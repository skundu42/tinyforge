import Testing

@testable import TinyForge

private func model(id: String, library: String? = "mlx", downloads: Int? = 1) -> HubModel {
    HubModel(
        id: id, author: id.split(separator: "/").first.map(String.init), downloads: downloads,
        likes: 0, gated: false, isPrivate: false, pipelineTag: "text-generation",
        libraryName: library, tags: [], lastModified: nil
    )
}

@MainActor
struct HubBrowserModelTests {
    @Test func searchPopulatesResults() async {
        let api = FakeBackendAPI()
        api.models = [model(id: "mlx-community/x-4bit")]
        let sut = HubBrowserModel(api: api, progress: FakeProgressStreaming(updates: []))
        sut.query = "x"

        await sut.search()

        #expect(sut.models.map(\.id) == ["mlx-community/x-4bit"])
    }

    @Test func searchFailureSetsFailedPhase() async {
        let api = FakeBackendAPI()
        api.searchError = APIError.unexpectedStatus(500)
        let sut = HubBrowserModel(api: api, progress: FakeProgressStreaming(updates: []))

        await sut.search()

        if case .failed = sut.phase {} else {
            Issue.record("expected failed phase, got \(sut.phase)")
        }
    }

    @Test func downloadTracksProgressToCompletion() async {
        let api = FakeBackendAPI()
        api.startResult = progress(repoId: "a/b", fraction: 0, state: "running")
        let updates = [
            progress(repoId: "a/b", fraction: 0.5, state: "running", downloaded: 50),
            progress(repoId: "a/b", fraction: 1.0, state: "completed", downloaded: 100),
        ]
        let sut = HubBrowserModel(api: api, progress: FakeProgressStreaming(updates: updates))

        await sut.download("a/b")

        #expect(sut.progress(for: "a/b")?.state == "completed")
        #expect(sut.progress(for: "a/b")?.fraction == 1.0)
    }

    @Test func downloadFailureRecordsErrorEntry() async {
        let api = FakeBackendAPI()  // startResult nil -> startDownload throws
        let sut = HubBrowserModel(api: api, progress: FakeProgressStreaming(updates: []))

        await sut.download("a/b")

        #expect(sut.progress(for: "a/b")?.state == "error")
    }
}
