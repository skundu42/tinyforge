import Foundation
import Observation

/// Drives the inference playground: pick a model + optional finetuned adapter,
/// enter a prompt, and stream base (and adapter) generations side by side.
@MainActor
@Observable
final class PlaygroundModel {
    var modelRepo = ""
    var prompt = "Write a haiku about the ocean."
    var adapterRunId = ""  // "" = base only
    var maxTokens = 200
    var temp = 0.7
    var topP = 0.9

    private(set) var cachedModels: [CachedRepo] = []
    private(set) var runs: [RunRecord] = []
    private(set) var baseOutput = ""
    private(set) var adapterOutput = ""
    private(set) var generating = false
    private(set) var error: String?

    private let api: any BackendAPI
    private let infer: any InferenceStreaming

    init(api: any BackendAPI, infer: any InferenceStreaming) {
        self.api = api
        self.infer = infer
    }

    var canGenerate: Bool { !modelRepo.isEmpty && !prompt.isEmpty && !generating }
    var hasAdapter: Bool { !adapterRunId.isEmpty }
    var selectedRun: RunRecord? { runs.first { $0.id == adapterRunId } }

    func loadInputs() async {
        cachedModels = ((try? await api.cacheInfo())?.repos ?? []).filter { $0.repoType == "model" }
        runs = ((try? await api.listRuns()) ?? []).filter { $0.state == "completed" }
    }

    func generate() async {
        generating = true
        error = nil
        baseOutput = ""
        adapterOutput = ""
        defer { generating = false }

        await runStream(adapterPath: nil) { [weak self] token in self?.baseOutput += token }
        if let run = selectedRun {
            await runStream(adapterPath: run.adapterPath) { [weak self] token in
                self?.adapterOutput += token
            }
        }
    }

    private func runStream(adapterPath: String?, append: @escaping @MainActor (String) -> Void) async {
        let request = GenRequest(
            modelRepo: modelRepo, adapterPath: adapterPath, prompt: prompt,
            maxTokens: maxTokens, temp: temp, topP: topP)
        for await event in infer.stream(request) {
            switch event {
            case .token(let text): append(text)
            case .failed(let message): error = message
            case .done: return
            }
        }
    }
}
