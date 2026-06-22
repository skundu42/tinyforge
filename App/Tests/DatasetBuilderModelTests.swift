import Testing

@testable import TinyForge

@MainActor
struct DatasetBuilderModelTests {
    @Test func loadPreviewPopulatesColumns() async {
        let api = FakeBackendAPI()
        api.preview = DatasetPreview(columns: ["instruction", "output"], rows: [], numRows: 0)
        let sut = DatasetBuilderModel(api: api)
        sut.repoId = "org/ds"

        await sut.loadPreview()

        #expect(sut.columns == ["instruction", "output"])
    }

    @Test func prepareRegistersAndRefreshesRegistry() async {
        let api = FakeBackendAPI()
        let sut = DatasetBuilderModel(api: api)
        sut.repoId = "org/ds"
        sut.datasetName = "math"

        await sut.loadPreview()
        await sut.prepare()

        #expect(api.preparedName == "math")
        #expect(sut.lastPrepared?.name == "math")
        #expect(sut.registered.contains { $0.name == "math" })
    }

    @Test func analyzePopulatesTokenStats() async {
        let api = FakeBackendAPI()
        let sut = DatasetBuilderModel(api: api)
        sut.tokenizerRepo = "org/tok"

        await sut.analyze()

        #expect(sut.tokenStats?.max == 2)
    }

    @Test func deleteRemovesFromRegistry() async {
        let api = FakeBackendAPI()
        let sut = DatasetBuilderModel(api: api)
        sut.repoId = "org/ds"
        sut.datasetName = "math"
        await sut.loadPreview()
        await sut.prepare()

        await sut.delete("ds1")

        #expect(api.deletedDataset == "ds1")
        #expect(sut.registered.isEmpty)
    }
}
