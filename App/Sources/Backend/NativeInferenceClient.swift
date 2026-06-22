import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Runs LLM generation natively in Swift via MLX-Swift — no Python round-trip.
/// Loads the model from the shared HuggingFace cache by repo id (the same cache
/// the Models tab downloads into), so a downloaded model runs offline.
///
/// Adapters aren't applied on this path yet; it generates from the base model.
struct NativeInferenceClient: InferenceStreaming {
    func stream(_ request: GenRequest) -> AsyncStream<InferEvent> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    let container = try await #huggingFaceLoadModelContainer(
                        configuration: ModelConfiguration(id: request.modelRepo))

                    let messages: [Chat.Message] = [.user(request.prompt)]
                    let input = try await container.prepare(input: UserInput(chat: messages))
                    let parameters = GenerateParameters(
                        maxTokens: request.maxTokens,
                        temperature: Float(request.temp),
                        topP: Float(request.topP))

                    let generation = try await container.generate(input: input, parameters: parameters)
                    for await event in generation {
                        if Task.isCancelled { break }
                        if case .chunk(let text) = event {
                            continuation.yield(.token(text))
                        }
                    }
                    continuation.yield(.done)
                    continuation.finish()
                } catch {
                    continuation.yield(.failed(String(describing: error)))
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
