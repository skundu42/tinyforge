import Foundation
import Testing

@testable import TinyForge

/// End-to-end: spawns the real Python backend from the repo's dev venv and
/// verifies the full handshake (ready line → REST). Self-locates the venv via
/// `#filePath`, so a plain `xcodebuild test` runs it locally; it skips when the
/// venv isn't present (e.g. CI without `uv sync`).
struct BackendIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // App
        .deletingLastPathComponent()  // repo root
    static let devPython = repoRoot.appending(path: "backend/.venv/bin/python3")
    static let backendDir = repoRoot.appending(path: "backend")

    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: devPython.path)))
    func backendSpawnsAndReportsHealthy() async throws {
        let token = TokenGenerator.make()
        let spec = LaunchSpec(
            executable: Self.devPython,
            arguments: BackendLauncher.arguments,
            workingDirectory: Self.backendDir,
            token: token,
            extraEnvironment: [:]
        )
        let manager = BackendProcessManager()

        let port = try await manager.start(spec, readyTimeout: .seconds(60))
        #expect(port > 0)
        #expect(manager.childPID != nil)

        let client = APIClient(
            baseURL: URL(string: "http://127.0.0.1:\(port)")!, token: token)
        var caught: Error?
        do {
            let health = try await client.health()
            #expect(health.status == "ok")
            #expect(health.name == "tinyforge")

            let runtime = try await client.runtime()
            #expect(runtime.platform == "darwin")
            #expect(runtime.machine == "arm64")
        } catch {
            caught = error
        }

        await manager.stop()
        #expect(manager.childPID == nil)
        if let caught { throw caught }
    }
}
