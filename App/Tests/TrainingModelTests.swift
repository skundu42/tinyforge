import Testing

@testable import TinyForge

@MainActor
struct TrainingModelTests {
    private func model(events: [TrainEvent] = []) -> (TrainingModel, FakeBackendAPI) {
        let api = FakeBackendAPI()
        let sut = TrainingModel(api: api, events: FakeRunEventStreaming(events: events))
        sut.name = "exp"
        sut.modelRepo = "mlx-community/x"
        sut.datasetId = "ds1"
        return (sut, api)
    }

    @Test func canStartRequiresModelDatasetAndName() {
        let (sut, _) = model()
        #expect(sut.canStart)
        sut.modelRepo = ""
        #expect(!sut.canStart)
    }

    @Test func torchEngineCanStartWithoutModelOrDataset() {
        let api = FakeBackendAPI()
        let sut = TrainingModel(api: api, events: FakeRunEventStreaming(events: []))
        sut.name = "scratch"
        sut.engine = "torch"  // from-scratch needs neither a model nor a dataset
        #expect(sut.canStart)
    }

    @Test func startSendsRequestAndIngestsMetrics() async {
        let events = [
            trainEvent("train", iter: 1, trainLoss: 5.0, tokensPerSec: 100, peakMemGb: 0.2, lr: 1e-5),
            trainEvent("val", iter: 1, valLoss: 4.5),
            trainEvent("train", iter: 2, trainLoss: 4.0, tokensPerSec: 120, peakMemGb: 0.2, lr: 1e-5),
            trainEvent("saved", path: "/runs/run1/adapters.safetensors"),
            trainEvent("status", state: "completed"),
        ]
        let (sut, api) = model(events: events)

        await sut.start()

        #expect(api.startedRequest?.name == "exp")
        #expect(sut.trainLoss.map(\.value) == [5.0, 4.0])
        #expect(sut.valLoss.map(\.value) == [4.5])
        #expect(sut.throughput.count == 2)
        #expect(sut.lastLR == 1e-5)
        #expect(sut.adapterPath == "/runs/run1/adapters.safetensors")
        #expect(sut.runState == "completed")
    }

    @Test func stopCallsApiAndMarksStopped() async {
        let (sut, api) = model()  // no events -> start() sets activeRunId then returns
        await sut.start()
        await sut.stop()
        #expect(api.stoppedRun == "run1")
        #expect(sut.runState == "stopped")
    }

    @Test func loadInputsFiltersCachedModelsAndLoadsDatasets() async {
        let (sut, api) = model()
        api.registered = [RegisteredDataset(
            id: "ds1", name: "d", targetFormat: "completion", trainRows: 8, valRows: 1,
            createdAt: "t", path: "/p")]
        api.cache = CacheInfo(sizeOnDisk: 0, repos: [
            CachedRepo(repoId: "m/x", repoType: "model", sizeOnDisk: 1, nbFiles: 1, lastAccessed: nil),
            CachedRepo(repoId: "d/y", repoType: "dataset", sizeOnDisk: 1, nbFiles: 1, lastAccessed: nil),
        ])

        await sut.loadInputs()

        #expect(sut.datasets.count == 1)
        #expect(sut.cachedModels.map(\.repoId) == ["m/x"])
    }
}
