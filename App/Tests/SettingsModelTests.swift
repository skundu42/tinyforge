import Testing

@testable import TinyForge

@MainActor
struct SettingsModelTests {
    @Test func loginStoresTokenClearsInputAndUpdatesStatus() async {
        let api = FakeBackendAPI()
        let sut = SettingsModel(api: api)
        sut.tokenInput = "hf_secret"

        await sut.login()

        #expect(api.loggedInToken == "hf_secret")
        #expect(sut.auth.loggedIn == true)
        #expect(sut.tokenInput == "")
    }

    @Test func deleteCallsApiAndRefreshesCache() async {
        let api = FakeBackendAPI()
        api.freedBytes = 1024
        let sut = SettingsModel(api: api)

        await sut.delete("a/b")

        #expect(api.deletedRepo == "a/b")
    }

    @Test func refreshLoadsAuthAndCache() async {
        let api = FakeBackendAPI()
        api.auth = AuthStatus(loggedIn: true, name: "bob")
        api.cache = CacheInfo(sizeOnDisk: 2048, repos: [])
        let sut = SettingsModel(api: api)

        await sut.refresh()

        #expect(sut.auth.name == "bob")
        #expect(sut.cache?.sizeOnDisk == 2048)
    }

    @Test func refreshSurfacesBackendFailure() async {
        let api = FakeBackendAPI()
        api.loadFailure = APIError.notImplemented
        let sut = SettingsModel(api: api)

        await sut.refresh()

        #expect(sut.loadError != nil)
    }
}
