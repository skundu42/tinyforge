import Testing

@testable import TinyForge

@MainActor
struct ExportModelTests {
    private func model() -> (ExportModel, FakeBackendAPI) {
        let api = FakeBackendAPI()
        let sut = ExportModel(api: api)
        sut.runId = "r1"
        return (sut, api)
    }

    @Test func startSendsRequestAndReportsSuccess() async {
        let (sut, api) = model()
        sut.target = "mlx"
        sut.qBits = 4

        await sut.start()

        #expect(api.startedExport?.runId == "r1")
        #expect(api.startedExport?.target == "mlx")
        #expect(sut.message?.contains("Exported to") == true)
    }

    @Test func startWithPushIncludesRepo() async {
        let (sut, api) = model()
        sut.pushRepo = "me/my-model"
        await sut.start()
        #expect(api.startedExport?.pushRepo == "me/my-model")
    }

    @Test func startReportsFailure() async {
        let (sut, api) = model()
        api.exportResult = ExportStatus(
            id: "exp1", runId: "r1", target: "safetensors", state: "failed",
            error: "fuse blew up", outputPath: nil, hubUrl: nil)

        await sut.start()

        #expect(sut.message?.contains("fuse blew up") == true)
    }

    @Test func loadInputsKeepsOnlyCompletedRuns() async {
        let (sut, api) = model()
        api.runs = [
            RunRecord(id: "r1", name: "a", modelRepo: "m", datasetId: "d", state: "completed", createdAt: "t", adapterPath: "/p"),
            RunRecord(id: "r2", name: "b", modelRepo: "m", datasetId: "d", state: "running", createdAt: "t", adapterPath: "/p"),
        ]
        await sut.loadInputs()
        #expect(sut.runs.map(\.id) == ["r1"])
    }

    @Test func loadInputsSurfacesBackendFailure() async {
        let (sut, api) = model()
        api.loadFailure = APIError.notImplemented

        await sut.loadInputs()

        #expect(sut.loadError != nil)
        #expect(sut.runs.isEmpty)
        #expect(sut.exports.isEmpty)
    }
}
