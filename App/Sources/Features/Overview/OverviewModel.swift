import Foundation
import Observation

/// Live counts that let the Home screen meet a beginner where they are.
@MainActor
@Observable
final class OverviewModel {
    private(set) var modelCount = 0
    private(set) var datasetCount = 0
    private(set) var finetuneCount = 0
    private(set) var loaded = false

    private let api: any BackendAPI

    init(api: any BackendAPI) {
        self.api = api
    }

    func load() async {
        let repos: [CachedRepo] = (try? await api.cacheInfo())?.repos ?? []
        modelCount = repos.filter { $0.repoType == "model" }.count

        let datasets: [RegisteredDataset] = (try? await api.listDatasets()) ?? []
        datasetCount = datasets.count

        let runs: [RunRecord] = (try? await api.listRuns()) ?? []
        finetuneCount = runs.filter { $0.state == "completed" }.count

        loaded = true
    }

    /// What the user has for a given workflow step (count, or nil if not a counted step).
    func count(for section: AppSection) -> Int? {
        switch section {
        case .hub: modelCount
        case .datasets: datasetCount
        case .train: finetuneCount
        default: nil
        }
    }
}
