import Foundation

/// Typed client for the backend REST API. The token is sent on `/v1` routes.
actor APIClient {
    let baseURL: URL
    let token: String
    private let transport: Transport
    private let decoder = JSONDecoder()

    init(baseURL: URL, token: String, transport: Transport = URLSessionTransport()) {
        self.baseURL = baseURL
        self.token = token
        self.transport = transport
    }

    func health() async throws -> HealthStatus {
        try await get("health", authorized: false)
    }

    func runtime() async throws -> RuntimeInfo {
        try await get("v1/runtime", authorized: true)
    }

    private func get<T: Decodable>(_ path: String, authorized: Bool) async throws -> T {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = "GET"
        if authorized {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await transport.send(request)
        switch response.statusCode {
        case 200:
            return try decoder.decode(T.self, from: data)
        case 401:
            throw APIError.unauthorized
        default:
            throw APIError.unexpectedStatus(response.statusCode)
        }
    }
}
