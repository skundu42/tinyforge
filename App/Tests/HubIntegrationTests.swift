import Foundation
import Testing

@testable import TinyForge

/// Full-stack M1 check: real Swift client → live backend → HuggingFace Hub,
/// including the WebSocket download-progress stream. Network-dependent, so it is
/// opt-in: create an empty `.run-network-tests` file in the repo root to enable.
struct HubIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let devPython = repoRoot.appending(path: "backend/.venv/bin/python3")
    static let backendDir = repoRoot.appending(path: "backend")

    static var enabled: Bool {
        FileManager.default.isExecutableFile(atPath: devPython.path)
            && FileManager.default.fileExists(atPath: repoRoot.appending(path: ".run-network-tests").path)
    }

    @Test(.enabled(if: enabled))
    func searchAndDownloadWithLiveProgress() async throws {
        let token = TokenGenerator.make()
        let spec = LaunchSpec(
            executable: Self.devPython, arguments: BackendLauncher.arguments,
            workingDirectory: Self.backendDir, token: token, extraEnvironment: [:]
        )
        let manager = BackendProcessManager()
        let port = try await manager.start(spec, readyTimeout: .seconds(60))
        let base = URL(string: "http://127.0.0.1:\(port)")!
        let api = APIClient(baseURL: base, token: token)
        let progressClient = DownloadProgressClient(baseURL: base, token: token)

        var caught: Error?
        do {
            // Live search through the backend to the Hub.
            let models = try await api.searchModels(query: "SmolLM", sort: "downloads", limit: 5)
            #expect(!models.isEmpty)

            // Start a tiny download and stream progress over the WebSocket.
            let started = try await api.startDownload(
                repoId: "mlx-community/SmolLM-135M-Instruct-4bit", repoType: "model")
            var terminal: DownloadProgress?
            for await update in progressClient.stream(jobId: started.id) {
                if update.isTerminal {
                    terminal = update
                    break
                }
            }
            #expect(terminal?.state == "completed")
            #expect(terminal?.fraction == 1.0)
        } catch {
            caught = error
        }

        await manager.stop()
        if let caught { throw caught }
    }
}
