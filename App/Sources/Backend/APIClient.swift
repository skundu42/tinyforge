import Foundation

/// Typed client for the backend REST API. Token is sent on every `/v1` route.
actor APIClient: BackendAPI {
    let baseURL: URL
    let token: String
    private let transport: Transport
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    init(baseURL: URL, token: String, transport: Transport = URLSessionTransport()) {
        self.baseURL = baseURL
        self.token = token
        self.transport = transport
    }

    // MARK: Core

    func health() async throws -> HealthStatus {
        try await request("GET", "health", authorized: false)
    }

    func runtime() async throws -> RuntimeInfo {
        try await request("GET", "v1/runtime")
    }

    // MARK: Hub

    func searchModels(query: String?, sort: String, limit: Int) async throws -> [HubModel] {
        try await request("GET", "v1/hub/models", query: searchQuery(query, sort, limit))
    }

    func searchDatasets(query: String?, sort: String, limit: Int) async throws -> [HubDataset] {
        try await request("GET", "v1/hub/datasets", query: searchQuery(query, sort, limit))
    }

    func modelDetail(repoId: String) async throws -> HubModelDetail {
        try await request("GET", "v1/hub/models/\(repoId)")
    }

    func startDownload(repoId: String, repoType: String) async throws -> DownloadProgress {
        try await request("POST", "v1/hub/downloads", body: ["repo_id": repoId, "repo_type": repoType])
    }

    func downloadProgress(id: String) async throws -> DownloadProgress {
        try await request("GET", "v1/hub/downloads/\(id)")
    }

    func cacheInfo() async throws -> CacheInfo {
        try await request("GET", "v1/hub/cache")
    }

    func deleteCached(repoId: String) async throws -> Int {
        let response: FreedResponse = try await request("DELETE", "v1/hub/cache/\(repoId)")
        return response.freedBytes
    }

    func authStatus() async throws -> AuthStatus {
        try await request("GET", "v1/hub/auth")
    }

    func login(token: String) async throws -> AuthStatus {
        try await request("POST", "v1/hub/auth/login", body: ["token": token])
    }

    func logout() async throws {
        let _: OkResponse = try await request("POST", "v1/hub/auth/logout")
    }

    // MARK: Transport

    private func searchQuery(_ query: String?, _ sort: String, _ limit: Int) -> [URLQueryItem] {
        var items = [URLQueryItem(name: "sort", value: sort), URLQueryItem(name: "limit", value: String(limit))]
        if let query, !query.isEmpty {
            items.append(URLQueryItem(name: "query", value: query))
        }
        return items
    }

    private func request<T: Decodable>(
        _ method: String,
        _ path: String,
        authorized: Bool = true,
        query: [URLQueryItem] = [],
        body: [String: String]? = nil
    ) async throws -> T {
        var components = URLComponents(
            url: baseURL.appending(path: path), resolvingAgainstBaseURL: false
        )!
        if !query.isEmpty {
            components.queryItems = query
        }

        var urlRequest = URLRequest(url: components.url!)
        urlRequest.httpMethod = method
        if authorized {
            urlRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            urlRequest.httpBody = try encoder.encode(body)
            urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await transport.send(urlRequest)
        switch response.statusCode {
        case 200..<300:
            return try decoder.decode(T.self, from: data)
        case 401:
            throw APIError.unauthorized
        default:
            throw APIError.unexpectedStatus(response.statusCode)
        }
    }
}

private struct FreedResponse: Decodable {
    let freedBytes: Int
    enum CodingKeys: String, CodingKey { case freedBytes = "freed_bytes" }
}

private struct OkResponse: Decodable {
    let ok: Bool
}
