import Foundation
import Testing

@testable import TinyForge

/// Native MLX-Swift generation from the shared HuggingFace cache (no Python).
/// Opt-in (loads a model, slow): create `.run-network-tests` and have the model
/// cached (download it in the Models tab first).
struct NativeInferenceIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static var enabled: Bool {
        FileManager.default.fileExists(atPath: repoRoot.appending(path: ".run-network-tests").path)
    }

    @Test(.enabled(if: enabled))
    func generatesNativelyFromCachedModel() async throws {
        let client = NativeInferenceClient()
        var output = ""
        var done = false
        let request = GenRequest(
            modelRepo: "mlx-community/SmolLM-135M-Instruct-4bit", adapterPath: nil,
            prompt: "Name one color.", maxTokens: 16, temp: 0.0)

        for await event in client.stream(request) {
            switch event {
            case .token(let text): output += text
            case .done: done = true
            case .failed(let message): Issue.record("native generation failed: \(message)")
            }
        }

        #expect(done)
        #expect(!output.isEmpty)
    }
}
