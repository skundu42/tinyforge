import Foundation

/// Streams training events from the backend's run WebSocket.
struct RunEventClient: RunEventStreaming {
    let baseURL: URL
    let token: String
    private let session: URLSession

    init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    private func socketURL(runId: String) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = (baseURL.scheme == "https") ? "wss" : "ws"
        components.path = "/v1/runs/\(runId)/ws"
        components.queryItems = [URLQueryItem(name: "token", value: token)]
        return components.url!
    }

    func stream(runId: String) -> AsyncStream<TrainEvent> {
        let url = socketURL(runId: runId)
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
                    if let data, let event = try? decoder.decode(TrainEvent.self, from: data) {
                        continuation.yield(event)
                        if event.event == "status" {
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
