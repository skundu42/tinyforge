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
        try await request(
            "POST", "v1/hub/downloads",
            bodyData: try encoder.encode(["repo_id": repoId, "repo_type": repoType]))
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
        try await request("POST", "v1/hub/auth/login", bodyData: try encoder.encode(["token": token]))
    }

    func logout() async throws {
        let _: OkResponse = try await request("POST", "v1/hub/auth/logout")
    }

    // MARK: Datasets

    func previewDataset(_ source: DatasetSource, limit: Int) async throws -> DatasetPreview {
        try await request("POST", "v1/datasets/preview", bodyData: try encoder.encode(PreviewBody(source: source, limit: limit)))
    }

    func analyzeDataset(source: DatasetSource, spec: FormatSpec, tokenizerRepo: String, sample: Int) async throws -> TokenStats {
        let body = AnalyzeBody(source: source, spec: spec, tokenizer_repo: tokenizerRepo, sample: sample)
        return try await request("POST", "v1/datasets/analyze", bodyData: try encoder.encode(body))
    }

    func prepareDataset(name: String, source: DatasetSource, spec: FormatSpec, valFraction: Double, seed: Int, maxRows: Int?) async throws -> RegisteredDataset {
        let body = PrepareBody(name: name, source: source, spec: spec, val_fraction: valFraction, seed: seed, max_rows: maxRows)
        return try await request("POST", "v1/datasets/prepare", bodyData: try encoder.encode(body))
    }

    func listDatasets() async throws -> [RegisteredDataset] {
        try await request("GET", "v1/datasets")
    }

    func deleteDataset(id: String) async throws {
        let _: OkResponse = try await request("DELETE", "v1/datasets/\(id)")
    }

    // MARK: Training runs

    func startRun(_ request: StartRunRequest) async throws -> RunRecord {
        try await self.request("POST", "v1/runs", bodyData: try encoder.encode(request))
    }

    func listRuns() async throws -> [RunRecord] {
        try await request("GET", "v1/runs")
    }

    func runStatus(id: String) async throws -> RunStatus {
        try await request("GET", "v1/runs/\(id)/status")
    }

    func stopRun(id: String) async throws {
        let _: OkResponse = try await request("POST", "v1/runs/\(id)/stop")
    }

    // MARK: Exports

    func startExport(_ request: ExportRequest) async throws -> ExportStatus {
        try await self.request("POST", "v1/exports", bodyData: try encoder.encode(request))
    }

    func listExports() async throws -> [ExportStatus] {
        try await request("GET", "v1/exports")
    }

    func getExport(id: String) async throws -> ExportStatus {
        try await request("GET", "v1/exports/\(id)")
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
        bodyData: Data? = nil
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
        if let bodyData {
            urlRequest.httpBody = bodyData
            urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await transport.send(urlRequest)
        switch response.statusCode {
        case 200..<300:
            return try decoder.decode(T.self, from: data)
        case 401:
            throw APIError.unauthorized
        default:
            throw APIError.unexpectedStatus(response.statusCode, detail: Self.errorDetail(from: data))
        }
    }

    /// Pull a human message out of a FastAPI error body: `{"detail": "..."}` or
    /// a 422 validation body `{"detail": [{"msg": "..."}]}`; else the raw text.
    private static func errorDetail(from data: Data) -> String? {
        guard !data.isEmpty else { return nil }
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let detail = object["detail"] as? String { return detail }
            if let items = object["detail"] as? [[String: Any]] {
                let messages = items.compactMap { $0["msg"] as? String }
                if !messages.isEmpty { return messages.joined(separator: "; ") }
            }
        }
        return String(data: data, encoding: .utf8)
    }
}

private struct FreedResponse: Decodable {
    let freedBytes: Int
    enum CodingKeys: String, CodingKey { case freedBytes = "freed_bytes" }
}

private struct OkResponse: Decodable {
    let ok: Bool
}

private struct PreviewBody: Encodable {
    let source: DatasetSource
    let limit: Int
}

private struct AnalyzeBody: Encodable {
    let source: DatasetSource
    let spec: FormatSpec
    let tokenizer_repo: String
    let sample: Int
}

private struct PrepareBody: Encodable {
    let name: String
    let source: DatasetSource
    let spec: FormatSpec
    let val_fraction: Double
    let seed: Int
    let max_rows: Int?
}
