import Foundation

@testable import TinyForge

/// Configurable fake backend for view-model tests.
final class FakeBackendAPI: BackendAPI, @unchecked Sendable {
    var models: [HubModel] = []
    var datasets: [HubDataset] = []
    var detail: HubModelDetail?
    var startResult: DownloadProgress?
    var cache = CacheInfo(sizeOnDisk: 0, repos: [])
    var auth = AuthStatus(loggedIn: false, name: nil)
    var searchError: Error?
    /// When set, the "load"/refresh methods (and `deleteCached`) throw it — used
    /// to exercise the error-surfacing paths that previously swallowed failures.
    var loadFailure: Error?
    var freedBytes = 0

    private(set) var loggedInToken: String?
    private(set) var deletedRepo: String?

    func health() async throws -> HealthStatus {
        HealthStatus(status: "ok", name: "tinyforge", version: "0")
    }

    func runtime() async throws -> RuntimeInfo {
        RuntimeInfo(pythonVersion: "3.13", platform: "darwin", machine: "arm64", engines: [:])
    }

    func searchModels(query: String?, sort: String, limit: Int) async throws -> [HubModel] {
        if let searchError { throw searchError }
        return models
    }

    func searchDatasets(query: String?, sort: String, limit: Int) async throws -> [HubDataset] {
        if let searchError { throw searchError }
        return datasets
    }

    func modelDetail(repoId: String) async throws -> HubModelDetail {
        guard let detail else { throw APIError.notImplemented }
        return detail
    }

    func startDownload(repoId: String, repoType: String) async throws -> DownloadProgress {
        guard let startResult else { throw APIError.notImplemented }
        return startResult
    }

    func downloadProgress(id: String) async throws -> DownloadProgress {
        guard let startResult else { throw APIError.notImplemented }
        return startResult
    }

    func cacheInfo() async throws -> CacheInfo {
        if let loadFailure { throw loadFailure }
        return cache
    }

    func deleteCached(repoId: String) async throws -> Int {
        if let loadFailure { throw loadFailure }
        deletedRepo = repoId
        return freedBytes
    }

    func authStatus() async throws -> AuthStatus {
        if let loadFailure { throw loadFailure }
        return auth
    }

    func login(token: String) async throws -> AuthStatus {
        loggedInToken = token
        auth = AuthStatus(loggedIn: true, name: "alice")
        return auth
    }

    func logout() async throws {
        auth = AuthStatus(loggedIn: false, name: nil)
    }

    // Datasets
    var preview = DatasetPreview(columns: ["text"], rows: [["text": .string("hi")]], numRows: 1)
    var tokenStats = TokenStats(count: 1, min: 1, max: 2, mean: 1.5, p50: 1, p95: 2, histogram: [])
    var registered: [RegisteredDataset] = []
    private(set) var preparedName: String?
    private(set) var deletedDataset: String?

    func previewDataset(_ source: DatasetSource, limit: Int) async throws -> DatasetPreview {
        preview
    }

    func analyzeDataset(source: DatasetSource, spec: FormatSpec, tokenizerRepo: String, sample: Int) async throws -> TokenStats {
        tokenStats
    }

    func prepareDataset(name: String, source: DatasetSource, spec: FormatSpec, valFraction: Double, seed: Int, maxRows: Int?) async throws -> RegisteredDataset {
        preparedName = name
        let record = RegisteredDataset(
            id: "ds1", name: name, targetFormat: "text", trainRows: 1, valRows: 0,
            createdAt: "t", path: "/data/ds1")
        registered.append(record)
        return record
    }

    func listDatasets() async throws -> [RegisteredDataset] {
        if let loadFailure { throw loadFailure }
        return registered
    }

    func deleteDataset(id: String) async throws {
        deletedDataset = id
        registered.removeAll { $0.id == id }
    }

    // Training
    var runs: [RunRecord] = []
    var startResultRun: RunRecord?
    private(set) var startedRequest: StartRunRequest?
    private(set) var stoppedRun: String?

    func startRun(_ request: StartRunRequest) async throws -> RunRecord {
        startedRequest = request
        let record = startResultRun ?? RunRecord(
            id: "run1", name: request.name, modelRepo: request.modelRepo,
            datasetId: request.datasetId, state: "running", createdAt: "t",
            adapterPath: "/runs/run1")
        runs.append(record)
        return record
    }

    func listRuns() async throws -> [RunRecord] {
        if let loadFailure { throw loadFailure }
        return runs
    }

    func runStatus(id: String) async throws -> RunStatus {
        RunStatus(id: id, name: "t", state: "running", error: nil, numEvents: 0)
    }

    func stopRun(id: String) async throws {
        stoppedRun = id
    }

    // Exports
    var exports: [ExportStatus] = []
    var exportResult: ExportStatus?
    private(set) var startedExport: ExportRequest?

    func startExport(_ request: ExportRequest) async throws -> ExportStatus {
        startedExport = request
        let status = exportResult ?? ExportStatus(
            id: "exp1", runId: request.runId, target: request.target, state: "completed",
            error: nil, outputPath: "/exports/exp1/fused", hubUrl: nil)
        exports.append(status)
        return status
    }

    func listExports() async throws -> [ExportStatus] {
        if let loadFailure { throw loadFailure }
        return exports
    }

    private(set) var getExportCalls = 0

    func getExport(id: String) async throws -> ExportStatus {
        getExportCalls += 1
        return exportResult ?? exports.first { $0.id == id } ?? ExportStatus(
            id: id, runId: "r1", target: "safetensors", state: "completed",
            error: nil, outputPath: "/exports/exp1/fused", hubUrl: nil)
    }
}

/// Thread-safe flag for observing async cancellation in tests.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set() { lock.withLock { value = true } }
    var isSet: Bool { lock.withLock { value } }
}

/// Inference stream that emits one token and then stays open until the consuming
/// task is cancelled, recording the cancellation.
struct CancelObservingInference: InferenceStreaming {
    let started: CancelFlag
    let cancelled: CancelFlag

    func stream(_ request: GenRequest) -> AsyncStream<InferEvent> {
        let started = started, cancelled = cancelled
        return AsyncStream { continuation in
            continuation.onTermination = { reason in
                if case .cancelled = reason { cancelled.set() }
            }
            started.set()
            continuation.yield(.token("partial"))
            // Never finish: emulate an in-progress generation.
        }
    }
}

/// Run-event stream that stays open until the consuming task is cancelled.
struct CancelObservingRunEvents: RunEventStreaming {
    let started: CancelFlag
    let cancelled: CancelFlag

    func stream(runId: String) -> AsyncStream<TrainEvent> {
        let started = started, cancelled = cancelled
        return AsyncStream { continuation in
            continuation.onTermination = { reason in
                if case .cancelled = reason { cancelled.set() }
            }
            started.set()
            continuation.yield(TrainEvent(
                event: "train", iter: 1, trainLoss: 5.0, valLoss: nil, lr: nil,
                tokensPerSec: nil, itPerSec: nil, peakMemGb: nil, trainedTokens: nil,
                state: nil, error: nil, path: nil, text: nil))
        }
    }
}

struct FakeRunEventStreaming: RunEventStreaming {
    let events: [TrainEvent]

    func stream(runId: String) -> AsyncStream<TrainEvent> {
        AsyncStream { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }
}

struct FakeInferenceStreaming: InferenceStreaming {
    let events: [InferEvent]
    func stream(_ request: GenRequest) -> AsyncStream<InferEvent> {
        AsyncStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }
}

func trainEvent(
    _ type: String, iter: Int? = nil, trainLoss: Double? = nil, valLoss: Double? = nil,
    tokensPerSec: Double? = nil, peakMemGb: Double? = nil, lr: Double? = nil,
    state: String? = nil, path: String? = nil
) -> TrainEvent {
    TrainEvent(
        event: type, iter: iter, trainLoss: trainLoss, valLoss: valLoss, lr: lr,
        tokensPerSec: tokensPerSec, itPerSec: nil, peakMemGb: peakMemGb, trainedTokens: nil,
        state: state, error: nil, path: path, text: nil)
}

/// Yields a fixed sequence of progress updates.
struct FakeProgressStreaming: ProgressStreaming {
    let updates: [DownloadProgress]

    func stream(jobId: String) -> AsyncStream<DownloadProgress> {
        AsyncStream { continuation in
            for update in updates {
                continuation.yield(update)
            }
            continuation.finish()
        }
    }
}

/// Progress stream that stays open until the consuming task is cancelled.
struct CancelObservingProgress: ProgressStreaming {
    let started: CancelFlag
    let cancelled: CancelFlag

    func stream(jobId: String) -> AsyncStream<DownloadProgress> {
        let started = started, cancelled = cancelled
        return AsyncStream { continuation in
            continuation.onTermination = { reason in
                if case .cancelled = reason { cancelled.set() }
            }
            started.set()
            continuation.yield(DownloadProgress(
                id: jobId, repoId: "m/x", repoType: "model", totalBytes: 100,
                downloadedBytes: 10, fraction: 0.1, state: "running", error: nil, localPath: nil))
        }
    }
}

func progress(
    repoId: String, fraction: Double, state: String, downloaded: Int = 0, total: Int = 100
) -> DownloadProgress {
    DownloadProgress(
        id: "job1", repoId: repoId, repoType: "model", totalBytes: total,
        downloadedBytes: downloaded, fraction: fraction, state: state, error: nil,
        localPath: state == "completed" ? "/cache/\(repoId)" : nil
    )
}
