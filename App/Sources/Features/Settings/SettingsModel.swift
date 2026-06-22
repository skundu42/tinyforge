import Foundation
import Observation

/// Drives the settings panel: HF token login and local cache management.
@MainActor
@Observable
final class SettingsModel: LoadErrorReporting {
    var tokenInput: String = ""
    private(set) var auth: AuthStatus = AuthStatus(loggedIn: false, name: nil)
    private(set) var cache: CacheInfo?
    private(set) var busy = false
    private(set) var message: String?
    var loadError: String?

    private let api: any BackendAPI

    init(api: any BackendAPI) {
        self.api = api
    }

    func refresh() async {
        await loadAuth()
        await loadCache()
    }

    func loadAuth() async {
        auth = await attempt("Load account") { try await api.authStatus() }
            ?? AuthStatus(loggedIn: false, name: nil)
    }

    func loadCache() async {
        cache = await attempt("Load cache") { try await api.cacheInfo() }
    }

    func login() async {
        guard !tokenInput.isEmpty else { return }
        busy = true
        defer { busy = false }
        do {
            auth = try await api.login(token: tokenInput)
            tokenInput = ""
            message = auth.loggedIn ? "Signed in as \(auth.name ?? "?")" : "Login failed"
        } catch {
            message = "Login failed: \(error)"
        }
    }

    func logout() async {
        await attempt("Sign out") { try await api.logout() }
        await loadAuth()
    }

    func delete(_ repoId: String) async {
        let freed = await attempt("Delete \(repoId)") { try await api.deleteCached(repoId: repoId) } ?? 0
        message = "Freed \(ByteFormat.string(freed))"
        await loadCache()
    }
}

enum ByteFormat {
    static func string(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
