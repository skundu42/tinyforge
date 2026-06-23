import Foundation

/// Abstraction over the network so APIClient can be tested without a live server.
protocol Transport: Sendable {
    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

struct URLSessionTransport: Transport {
    let session: URLSession

    /// A bounded session: a request idle for `timeoutIntervalForRequest`, or a
    /// whole operation past `timeoutIntervalForResource`, fails instead of hanging
    /// forever (URLSession's resource default is 7 days). The request timeout is
    /// generous enough for slow dataset loads/exports.
    static func makeDefaultSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 3600
        return URLSession(configuration: config)
    }

    init(session: URLSession) {
        self.session = session
    }

    init() {
        self.session = URLSessionTransport.makeDefaultSession()
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
    /// A non-2xx response, carrying the backend's error detail (FastAPI `detail`)
    /// when present so the UI can show why instead of an opaque status code.
    case unexpectedStatus(Int, detail: String?)
    case notImplemented
}

extension APIError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .nonHTTPResponse:
            return "The backend returned an invalid response."
        case .unauthorized:
            return "Not authorized to reach the backend."
        case .unexpectedStatus(let code, let detail):
            if let detail, !detail.isEmpty { return detail }
            return "The backend returned an error (HTTP \(code))."
        case .notImplemented:
            return "This feature isn't available yet."
        }
    }
}
