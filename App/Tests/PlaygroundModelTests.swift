import Testing

@testable import TinyForge

@MainActor
struct PlaygroundModelTests {
    private func model(events: [InferEvent]) -> (PlaygroundModel, FakeBackendAPI) {
        let api = FakeBackendAPI()
        let sut = PlaygroundModel(api: api, infer: FakeInferenceStreaming(events: events))
        sut.modelRepo = "mlx-community/x"
        sut.prompt = "hi"
        return (sut, api)
    }

    @Test func generateStreamsBaseOutput() async {
        let (sut, _) = model(events: [.token("Hel"), .token("lo"), .done])
        await sut.generate()
        #expect(sut.baseOutput == "Hello")
        #expect(sut.adapterOutput.isEmpty)
    }

    @Test func generateAlsoStreamsAdapterWhenSelected() async {
        let (sut, api) = model(events: [.token("out"), .done])
        api.runs = [RunRecord(
            id: "run1", name: "ft", modelRepo: "mlx-community/x", datasetId: "ds1",
            state: "completed", createdAt: "t", adapterPath: "/runs/run1")]
        await sut.loadInputs()
        sut.adapterRunId = "run1"

        await sut.generate()

        #expect(sut.baseOutput == "out")
        #expect(sut.adapterOutput == "out")  // adapter stream uses same fake events
    }

    @Test func generateSurfacesError() async {
        let (sut, _) = model(events: [.failed("model not found")])
        await sut.generate()
        #expect(sut.error == "model not found")
    }

    @Test func loadInputsFiltersModelsAndCompletedRuns() async {
        let (sut, api) = model(events: [])
        api.cache = CacheInfo(sizeOnDisk: 0, repos: [
            CachedRepo(repoId: "m/x", repoType: "model", sizeOnDisk: 1, nbFiles: 1, lastAccessed: nil)])
        api.runs = [
            RunRecord(id: "r1", name: "done", modelRepo: "m", datasetId: "d", state: "completed", createdAt: "t", adapterPath: "/p"),
            RunRecord(id: "r2", name: "fail", modelRepo: "m", datasetId: "d", state: "failed", createdAt: "t", adapterPath: "/p"),
        ]

        await sut.loadInputs()

        #expect(sut.cachedModels.count == 1)
        #expect(sut.runs.map(\.id) == ["r1"])
    }

    @Test func loadInputsSurfacesBackendFailure() async {
        let (sut, api) = model(events: [])
        api.loadFailure = APIError.notImplemented

        await sut.loadInputs()

        #expect(sut.loadError != nil)
        #expect(sut.cachedModels.isEmpty)
        #expect(sut.runs.isEmpty)
    }

    @Test func scratchRunsListsOnlyLmEngineRuns() async {
        let api = FakeBackendAPI()
        api.runs = [
            runRecord(id: "a", state: "completed", engine: "lm", adapterPath: "/runs/a"),
            runRecord(id: "b", state: "completed", engine: "mlx", adapterPath: "/runs/b"),
        ]
        let sut = PlaygroundModel(api: api, infer: FakeInferenceStreaming(events: []))
        await sut.loadInputs()
        #expect(sut.scratchRuns.map(\.id) == ["a"])
    }

    @Test func selectScratchModelPointsModelRepoAtRunDir() {
        let sut = PlaygroundModel(api: FakeBackendAPI(), infer: FakeInferenceStreaming(events: []))
        let run = runRecord(id: "a", state: "completed", engine: "lm", adapterPath: "/runs/a")
        sut.selectScratchModel(run)
        #expect(sut.modelRepo == "/runs/a")
        #expect(sut.adapterRunId == "")
    }
}

private func runRecord(
    id: String, name: String = "run", modelRepo: String = "m", datasetId: String = "d",
    state: String, engine: String = "mlx", createdAt: String = "t", adapterPath: String
) -> RunRecord {
    RunRecord(id: id, name: name, modelRepo: modelRepo, datasetId: datasetId,
              state: state, engine: engine, createdAt: createdAt, adapterPath: adapterPath)
}
