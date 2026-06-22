import Foundation

/// Abstraction over the network so APIClient can be tested without a live server.
protocol Transport: Sendable {
    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

struct URLSessionTransport: Transport {
    let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.nonHTTPResponse
        }
        return (data, http)
    }
}

enum APIError: Error, Equatable {
    case nonHTTPResponse
    case unauthorized
    case unexpectedStatus(Int)
    case notImplemented
}
