import Testing

@testable import TinyForge

@MainActor
struct OverviewModelTests {
    @Test func loadCountsModelsDatasetsAndCompletedRuns() async {
        let api = FakeBackendAPI()
        api.cache = CacheInfo(sizeOnDisk: 0, repos: [
            CachedRepo(repoId: "m/x", repoType: "model", sizeOnDisk: 1, nbFiles: 1, lastAccessed: nil),
            CachedRepo(repoId: "d/y", repoType: "dataset", sizeOnDisk: 1, nbFiles: 1, lastAccessed: nil),
        ])
        api.registered = [RegisteredDataset(
            id: "d1", name: "a", targetFormat: "text", trainRows: 1, valRows: 0,
            createdAt: "t", path: "/p")]
        api.runs = [
            RunRecord(id: "r1", name: "a", modelRepo: "m", datasetId: "d", state: "completed", createdAt: "t", adapterPath: "/p"),
            RunRecord(id: "r2", name: "b", modelRepo: "m", datasetId: "d", state: "running", createdAt: "t", adapterPath: "/p"),
        ]

        let sut = OverviewModel(api: api)
        await sut.load()

        #expect(sut.modelCount == 1)  // only the "model" repo, not the dataset repo
        #expect(sut.datasetCount == 1)
        #expect(sut.finetuneCount == 1)  // only the completed run
        #expect(sut.count(for: .hub) == 1)
    }

    @Test func loadSurfacesBackendFailureAndDegradesGracefully() async {
        let api = FakeBackendAPI()
        api.loadFailure = APIError.notImplemented
        let sut = OverviewModel(api: api)

        await sut.load()

        #expect(sut.loadError != nil)
        #expect(sut.modelCount == 0)
        #expect(sut.datasetCount == 0)
        #expect(sut.finetuneCount == 0)
    }
}
