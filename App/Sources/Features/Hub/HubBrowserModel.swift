import Foundation
import Observation

/// Drives the Hub browser: search, model detail, and downloads with live progress.
@MainActor
@Observable
final class HubBrowserModel: LoadErrorReporting {
    enum Phase: Equatable {
        case idle
        case searching
        case results([HubModel])
        case failed(String)
    }

    var query: String = ""
    var sort: String = "downloads"
    private(set) var phase: Phase = .idle
    private(set) var selectedDetail: HubModelDetail?
    private(set) var detailLoading = false
    private(set) var downloads: [String: DownloadProgress] = [:]
    private(set) var downloadedModels: [CachedRepo] = []
    var loadError: String?

    private let api: any BackendAPI
    private let progress: any ProgressStreaming
    private var downloadTasks: [String: Task<Void, Never>] = [:]

    init(api: any BackendAPI, progress: any ProgressStreaming) {
        self.api = api
        self.progress = progress
    }

    var models: [HubModel] {
        if case .results(let models) = phase { return models }
        return []
    }

    func search() async {
        phase = .searching
        do {
            let results = try await api.searchModels(
                query: query.isEmpty ? nil : query, sort: sort, limit: 40
            )
            phase = .results(results)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func loadDetail(_ repoId: String) async {
        detailLoading = true
        selectedDetail = nil
        defer { detailLoading = false }
        selectedDetail = await attempt("Load model details") { try await api.modelDetail(repoId: repoId) }
    }

    func loadDownloaded() async {
        let repos = (await attempt("Load downloads") { try await api.cacheInfo() })?.repos ?? []
        downloadedModels = repos.filter { $0.repoType == "model" }
    }

    func deleteDownloaded(_ repoId: String) async {
        await attempt("Delete \(repoId)") { try await api.deleteCached(repoId: repoId) }
        await loadDownloaded()
    }

    /// Starts a download as a cancellable task so leaving the screen tears the
    /// progress stream down instead of leaking it.
    func startDownload(_ repoId: String) {
        downloadTasks[repoId]?.cancel()
        downloadTasks[repoId] = Task { [weak self] in await self?.download(repoId) }
    }

    /// Cancel all in-flight download progress streams (e.g. on view disappear).
    func cancelStreaming() {
        for task in downloadTasks.values { task.cancel() }
        downloadTasks.removeAll()
    }

    func download(_ repoId: String) async {
        do {
            let started = try await api.startDownload(repoId: repoId, repoType: "model")
            downloads[repoId] = started
            for await update in progress.stream(jobId: started.id) {
                downloads[repoId] = update
                if update.isTerminal { break }
            }
            await loadDownloaded()  // refresh the downloaded list after a finished download
        } catch {
            downloads[repoId] = DownloadProgress(
                id: "", repoId: repoId, repoType: "model", totalBytes: 0,
                downloadedBytes: 0, fraction: 0, state: "error",
                error: error.localizedDescription, localPath: nil
            )
        }
    }

    func progress(for repoId: String) -> DownloadProgress? {
        downloads[repoId]
    }
}
