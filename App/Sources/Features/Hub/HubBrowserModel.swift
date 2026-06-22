import Foundation
import Observation

/// Drives the Hub browser: search, model detail, and downloads with live progress.
@MainActor
@Observable
final class HubBrowserModel {
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

    private let api: any BackendAPI
    private let progress: any ProgressStreaming

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
            phase = .failed(String(describing: error))
        }
    }

    func loadDetail(_ repoId: String) async {
        detailLoading = true
        selectedDetail = nil
        defer { detailLoading = false }
        selectedDetail = try? await api.modelDetail(repoId: repoId)
    }

    func loadDownloaded() async {
        let repos = (try? await api.cacheInfo())?.repos ?? []
        downloadedModels = repos.filter { $0.repoType == "model" }
    }

    func deleteDownloaded(_ repoId: String) async {
        try? await api.deleteCached(repoId: repoId)
        await loadDownloaded()
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
                error: String(describing: error), localPath: nil
            )
        }
    }

    func progress(for repoId: String) -> DownloadProgress? {
        downloads[repoId]
    }
}
