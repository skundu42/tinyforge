import Foundation
import Testing

@testable import TinyForge

/// Full-stack M4 check: real Swift InferenceClient → backend → mlx_lm streaming
/// generation over the WebSocket. Uses the cached model offline. Opt-in (slow):
/// create `.run-network-tests`.
struct InferenceIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let devPython = repoRoot.appending(path: "backend/.venv/bin/python3")
    static let backendDir = repoRoot.appending(path: "backend")

    static var enabled: Bool {
        FileManager.default.isExecutableFile(atPath: devPython.path)
            && FileManager.default.fileExists(atPath: repoRoot.appending(path: ".run-network-tests").path)
    }

    @Test(.enabled(if: enabled))
    func streamsGeneratedTokens() async throws {
        let token = TokenGenerator.make()
        let spec = LaunchSpec(
            executable: Self.devPython, arguments: BackendLauncher.arguments,
            workingDirectory: Self.backendDir, token: token,
            extraEnvironment: ["HF_HUB_OFFLINE": "1"]
        )
        let manager = BackendProcessManager()
        let port = try await manager.start(spec, readyTimeout: .seconds(60))
        let client = InferenceClient(
            baseURL: URL(string: "http://127.0.0.1:\(port)")!, token: token)

        var output = ""
        var done = false
        let request = GenRequest(
            modelRepo: "mlx-community/SmolLM-135M-Instruct-4bit", adapterPath: nil,
            prompt: "Name one color.", maxTokens: 16, temp: 0.0)
        for await event in client.stream(request) {
            switch event {
            case .token(let text): output += text
            case .done: done = true
            case .failed(let message): Issue.record("generation failed: \(message)")
            }
        }

        await manager.stop()
        #expect(done)
        #expect(!output.isEmpty)
    }
}
