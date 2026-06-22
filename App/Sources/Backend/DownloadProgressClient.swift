import Foundation

/// Streams download progress updates for a job. Abstracted so view models can
/// be tested with a fake stream instead of a live WebSocket.
protocol ProgressStreaming: Sendable {
    func stream(jobId: String) -> AsyncStream<DownloadProgress>
}

/// Live implementation backed by the backend's WebSocket endpoint.
struct DownloadProgressClient: ProgressStreaming {
    let baseURL: URL
    let token: String
    private let session: URLSession

    init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    private func socketURL(jobId: String) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = (baseURL.scheme == "https") ? "wss" : "ws"
        components.path = "/v1/hub/downloads/\(jobId)/ws"
        components.queryItems = [URLQueryItem(name: "token", value: token)]
        return components.url!
    }

    func stream(jobId: String) -> AsyncStream<DownloadProgress> {
        let url = socketURL(jobId: jobId)
        let session = session
        return AsyncStream { continuation in
            let task = session.webSocketTask(with: url)
            task.resume()

            let receiver = Task {
                let decoder = JSONDecoder()
                defer { task.cancel(with: .goingAway, reason: nil) }
                while !Task.isCancelled {
                    let message: URLSessionWebSocketTask.Message
                    do {
                        message = try await task.receive()
                    } catch {
                        continuation.finish()
                        return
                    }
                    let data: Data? =
                        switch message {
                        case .string(let text): Data(text.utf8)
                        case .data(let raw): raw
                        @unknown default: nil
                        }
                    if let data, let progress = try? decoder.decode(DownloadProgress.self, from: data) {
                        continuation.yield(progress)
                        if progress.isTerminal {
                            continuation.finish()
                            return
                        }
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                receiver.cancel()
                task.cancel(with: .goingAway, reason: nil)
            }
        }
    }
}
