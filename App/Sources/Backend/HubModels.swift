import Foundation

struct HubModel: Codable, Identifiable, Sendable, Equatable {
    let id: String
    let author: String?
    let downloads: Int?
    let likes: Int?
    let gated: Bool
    let isPrivate: Bool
    let pipelineTag: String?
    let libraryName: String?
    let tags: [String]
    let lastModified: String?

    enum CodingKeys: String, CodingKey {
        case id, author, downloads, likes, gated, tags
        case isPrivate = "private"
        case pipelineTag = "pipeline_tag"
        case libraryName = "library_name"
        case lastModified = "last_modified"
    }
}

struct HubDataset: Codable, Identifiable, Sendable, Equatable {
    let id: String
    let author: String?
    let downloads: Int?
    let likes: Int?
    let gated: Bool
    let isPrivate: Bool
    let tags: [String]
    let lastModified: String?

    enum CodingKeys: String, CodingKey {
        case id, author, downloads, likes, gated, tags
        case isPrivate = "private"
        case lastModified = "last_modified"
    }
}

struct HubFile: Codable, Identifiable, Sendable, Equatable {
    var id: String { filename }
    let filename: String
    let size: Int?
}

struct HubModelDetail: Codable, Identifiable, Sendable, Equatable {
    let id: String
    let author: String?
    let downloads: Int?
    let likes: Int?
    let gated: Bool
    let pipelineTag: String?
    let libraryName: String?
    let tags: [String]
    let siblings: [HubFile]
    let totalSize: Int?
    let readme: String?

    enum CodingKeys: String, CodingKey {
        case id, author, downloads, likes, gated, tags, siblings, readme
        case pipelineTag = "pipeline_tag"
        case libraryName = "library_name"
        case totalSize = "total_size"
    }
}

struct DownloadProgress: Codable, Identifiable, Sendable, Equatable {
    let id: String
    let repoId: String
    let repoType: String
    let totalBytes: Int
    let downloadedBytes: Int
    let fraction: Double
    let state: String
    let error: String?
    let localPath: String?

    var isTerminal: Bool { state == "completed" || state == "error" }

    enum CodingKeys: String, CodingKey {
        case id, fraction, state, error
        case repoId = "repo_id"
        case repoType = "repo_type"
        case totalBytes = "total_bytes"
        case downloadedBytes = "downloaded_bytes"
        case localPath = "local_path"
    }
}

struct CachedRepo: Codable, Identifiable, Sendable, Equatable {
    var id: String { repoId }
    let repoId: String
    let repoType: String
    let sizeOnDisk: Int
    let nbFiles: Int
    let lastAccessed: Double?

    enum CodingKeys: String, CodingKey {
        case repoId = "repo_id"
        case repoType = "repo_type"
        case sizeOnDisk = "size_on_disk"
        case nbFiles = "nb_files"
        case lastAccessed = "last_accessed"
    }
}

struct CacheInfo: Codable, Sendable, Equatable {
    let sizeOnDisk: Int
    let repos: [CachedRepo]

    enum CodingKeys: String, CodingKey {
        case sizeOnDisk = "size_on_disk"
        case repos
    }
}

struct AuthStatus: Codable, Sendable, Equatable {
    let loggedIn: Bool
    let name: String?

    enum CodingKeys: String, CodingKey {
        case loggedIn = "logged_in"
        case name
    }
}

/// The backend surface the UI depends on. APIClient is the live implementation;
/// view models take `any BackendAPI` so they can be tested with fakes.
protocol BackendAPI: Sendable {
    func health() async throws -> HealthStatus
    func runtime() async throws -> RuntimeInfo
    func searchModels(query: String?, sort: String, limit: Int) async throws -> [HubModel]
    func searchDatasets(query: String?, sort: String, limit: Int) async throws -> [HubDataset]
    func modelDetail(repoId: String) async throws -> HubModelDetail
    func startDownload(repoId: String, repoType: String) async throws -> DownloadProgress
    func downloadProgress(id: String) async throws -> DownloadProgress
    func cacheInfo() async throws -> CacheInfo
    func deleteCached(repoId: String) async throws -> Int
    func authStatus() async throws -> AuthStatus
    func login(token: String) async throws -> AuthStatus
    func logout() async throws
    func previewDataset(_ source: DatasetSource, limit: Int) async throws -> DatasetPreview
    func analyzeDataset(source: DatasetSource, spec: FormatSpec, tokenizerRepo: String, sample: Int) async throws -> TokenStats
    func prepareDataset(name: String, source: DatasetSource, spec: FormatSpec, valFraction: Double, seed: Int, maxRows: Int?) async throws -> RegisteredDataset
    func listDatasets() async throws -> [RegisteredDataset]
    func deleteDataset(id: String) async throws
}
