import Foundation
import Observation

/// Drives the dataset builder: source → preview → format → (token analysis) →
/// prepare → registry.
@MainActor
@Observable
final class DatasetBuilderModel: LoadErrorReporting {
    // Source
    var sourceKind = "hub"  // hub | local
    var repoId = ""
    var localPath = ""
    var split = "train"

    // Preview
    private(set) var preview: DatasetPreview?
    private(set) var previewError: String?
    private(set) var loadingPreview = false

    // Format
    var spec = FormatSpec()

    // Token analysis
    var tokenizerRepo = ""
    private(set) var tokenStats: TokenStats?
    private(set) var analyzing = false

    // Prepare
    var datasetName = ""
    var valFraction = 0.1
    private(set) var preparing = false
    private(set) var lastPrepared: RegisteredDataset?
    private(set) var message: String?

    // Registry
    private(set) var registered: [RegisteredDataset] = []

    var loadError: String?

    private let api: any BackendAPI

    init(api: any BackendAPI) {
        self.api = api
    }

    var source: DatasetSource {
        if sourceKind == "hub" {
            DatasetSource(kind: "hub", repoId: repoId.isEmpty ? nil : repoId, split: split)
        } else {
            DatasetSource(kind: "local", path: localPath.isEmpty ? nil : localPath, split: split)
        }
    }

    var columns: [String] { preview?.columns ?? [] }
    var canPreview: Bool { sourceKind == "hub" ? !repoId.isEmpty : !localPath.isEmpty }
    var canPrepare: Bool { preview != nil && !datasetName.isEmpty }

    func loadPreview() async {
        loadingPreview = true
        previewError = nil
        defer { loadingPreview = false }
        do {
            preview = try await api.previewDataset(source, limit: 25)
        } catch {
            previewError = error.localizedDescription
        }
    }

    func analyze() async {
        guard !tokenizerRepo.isEmpty else { return }
        analyzing = true
        defer { analyzing = false }
        tokenStats = await attempt("Analyze tokens") {
            try await api.analyzeDataset(
                source: source, spec: spec, tokenizerRepo: tokenizerRepo, sample: 200)
        }
    }

    func prepare() async {
        preparing = true
        defer { preparing = false }
        do {
            let record = try await api.prepareDataset(
                name: datasetName, source: source, spec: spec,
                valFraction: valFraction, seed: 0, maxRows: nil)
            lastPrepared = record
            message = "Prepared “\(record.name)” — \(record.trainRows) train / \(record.valRows) val rows"
            await loadRegistry()
        } catch {
            message = "Prepare failed: \(error.localizedDescription)"
        }
    }

    func loadRegistry() async {
        registered = await attempt("Load datasets") { try await api.listDatasets() } ?? []
    }

    func delete(_ id: String) async {
        await attempt("Delete dataset") { try await api.deleteDataset(id: id) }
        await loadRegistry()
    }
}
