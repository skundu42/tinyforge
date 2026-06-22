import Foundation

/// Streams generated tokens from the backend's inference WebSocket. Sends the
/// request as the first message, then yields token / done / error events.
struct InferenceClient: InferenceStreaming {
    let baseURL: URL
    let token: String
    private let session: URLSession

    init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    private var socketURL: URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = (baseURL.scheme == "https") ? "wss" : "ws"
        components.path = "/v1/infer/ws"
        components.queryItems = [URLQueryItem(name: "token", value: token)]
        return components.url!
    }

    func stream(_ request: GenRequest) -> AsyncStream<InferEvent> {
        let url = socketURL
        let session = session
        return AsyncStream { continuation in
            let task = session.webSocketTask(with: url)
            task.resume()

            let receiver = Task {
                let decoder = JSONDecoder()
                defer { task.cancel(with: .goingAway, reason: nil) }

                // Send the request first.
                if let data = try? JSONEncoder().encode(request),
                    let json = String(data: data, encoding: .utf8) {
                    do { try await task.send(.string(json)) } catch {
                        continuation.finish()
                        return
                    }
                }

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
                    guard let data, let envelope = try? decoder.decode(InferMessage.self, from: data)
                    else { continue }

                    switch envelope.event {
                    case "token":
                        continuation.yield(.token(envelope.text ?? ""))
                    case "error":
                        continuation.yield(.failed(envelope.error ?? "error"))
                        continuation.finish()
                        return
                    default:  // "done"
                        continuation.yield(.done)
                        continuation.finish()
                        return
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

private struct InferMessage: Decodable {
    let event: String
    let text: String?
    let error: String?
}
