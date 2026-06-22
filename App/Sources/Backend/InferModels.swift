import Foundation

struct GenRequest: Codable, Sendable {
    var modelRepo: String
    var adapterPath: String?
    var prompt: String
    var maxTokens: Int = 256
    var temp: Double = 0.7
    var topP: Double = 0.9
    var chat: Bool = true

    enum CodingKeys: String, CodingKey {
        case prompt, temp, chat
        case modelRepo = "model_repo"
        case adapterPath = "adapter_path"
        case maxTokens = "max_tokens"
        case topP = "top_p"
    }
}

enum InferEvent: Sendable, Equatable {
    case token(String)
    case done
    case failed(String)
}

/// Streams generated tokens for a request. Abstracted so the playground view
/// model can be tested with a fake.
protocol InferenceStreaming: Sendable {
    func stream(_ request: GenRequest) -> AsyncStream<InferEvent>
}
