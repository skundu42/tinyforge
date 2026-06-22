import Foundation
import Observation

/// Drives the inference playground: pick a model + optional finetuned adapter,
/// enter a prompt, and stream base (and adapter) generations side by side.
@MainActor
@Observable
final class PlaygroundModel: LoadErrorReporting {
    var modelRepo = ""
    var prompt = "Write a haiku about the ocean."
    var adapterRunId = ""  // "" = base only
    var maxTokens = 200
    var temp = 0.7
    var topP = 0.9
    var runNative = false  // run in Swift via MLX-Swift instead of the Python backend

    private(set) var cachedModels: [CachedRepo] = []
    private(set) var runs: [RunRecord] = []
    private(set) var baseOutput = ""
    private(set) var adapterOutput = ""
    private(set) var generating = false
    private(set) var error: String?
    var loadError: String?

    private let api: any BackendAPI
    private let infer: any InferenceStreaming
    private let nativeInfer: any InferenceStreaming

    init(
        api: any BackendAPI, infer: any InferenceStreaming,
        nativeInfer: (any InferenceStreaming)? = nil
    ) {
        self.api = api
        self.infer = infer
        self.nativeInfer = nativeInfer ?? infer
    }

    var canGenerate: Bool { !modelRepo.isEmpty && !prompt.isEmpty && !generating }
    var hasAdapter: Bool { !adapterRunId.isEmpty }
    // Native runs base-only; the side-by-side comparison uses the backend path.
    var showsComparison: Bool { hasAdapter && !runNative }
    var selectedRun: RunRecord? { runs.first { $0.id == adapterRunId } }

    func loadInputs() async {
        let repos = (await attempt("Load models") { try await api.cacheInfo() })?.repos ?? []
        cachedModels = repos.filter { $0.repoType == "model" }
        runs = (await attempt("Load finetunes") { try await api.listRuns() } ?? [])
            .filter { $0.state == "completed" }
    }

    func generate() async {
        generating = true
        error = nil
        baseOutput = ""
        adapterOutput = ""
        defer { generating = false }

        let baseStreamer: any InferenceStreaming = runNative ? nativeInfer : infer
        await runStream(streamer: baseStreamer, adapterPath: nil) { [weak self] token in
            self?.baseOutput += token
        }
        if !runNative, let run = selectedRun {
            await runStream(streamer: infer, adapterPath: run.adapterPath) { [weak self] token in
                self?.adapterOutput += token
            }
        }
    }

    private func runStream(
        streamer: any InferenceStreaming, adapterPath: String?,
        append: @escaping @MainActor (String) -> Void
    ) async {
        let request = GenRequest(
            modelRepo: modelRepo, adapterPath: adapterPath, prompt: prompt,
            maxTokens: maxTokens, temp: temp, topP: topP)
        for await event in streamer.stream(request) {
            switch event {
            case .token(let text): append(text)
            case .failed(let message): error = message
            case .done: return
            }
        }
    }
}
