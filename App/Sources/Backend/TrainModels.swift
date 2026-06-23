import Foundation

struct StartRunRequest: Codable, Sendable {
    var name: String
    var modelRepo: String
    var datasetId: String
    var engine: String = "mlx"
    var fineTuneType: String = "lora"
    var numLayers: Int = 16
    var batchSize: Int = 1
    var iters: Int = 100
    var learningRate: Double = 1e-5
    var stepsPerReport: Int = 10
    var stepsPerEval: Int = 50
    var maxSeqLength: Int = 512
    var gradCheckpoint: Bool = true
    var seed: Int = 0
    var modelSize: String = "small"
    var hiddenSize: Int = 256
    var numHeads: Int = 8
    var vocabSize: Int = 8000
    var contextLength: Int = 512

    enum CodingKeys: String, CodingKey {
        case name, iters, seed, engine
        case modelRepo = "model_repo"
        case datasetId = "dataset_id"
        case fineTuneType = "fine_tune_type"
        case numLayers = "num_layers"
        case batchSize = "batch_size"
        case learningRate = "learning_rate"
        case stepsPerReport = "steps_per_report"
        case stepsPerEval = "steps_per_eval"
        case maxSeqLength = "max_seq_length"
        case gradCheckpoint = "grad_checkpoint"
        case modelSize = "model_size"
        case hiddenSize = "hidden_size"
        case numHeads = "num_heads"
        case vocabSize = "vocab_size"
        case contextLength = "context_length"
    }
}

struct RunRecord: Codable, Sendable, Identifiable, Equatable {
    let id: String
    let name: String
    let modelRepo: String
    let datasetId: String
    let state: String
    let engine: String
    let createdAt: String
    let adapterPath: String

    enum CodingKeys: String, CodingKey {
        case id, name, state, engine
        case modelRepo = "model_repo"
        case datasetId = "dataset_id"
        case createdAt = "created_at"
        case adapterPath = "adapter_path"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        modelRepo = try c.decode(String.self, forKey: .modelRepo)
        datasetId = try c.decode(String.self, forKey: .datasetId)
        state = try c.decode(String.self, forKey: .state)
        engine = try c.decodeIfPresent(String.self, forKey: .engine) ?? "mlx"
        createdAt = try c.decode(String.self, forKey: .createdAt)
        adapterPath = try c.decode(String.self, forKey: .adapterPath)
    }

    init(id: String, name: String, modelRepo: String, datasetId: String, state: String,
         engine: String = "mlx", createdAt: String, adapterPath: String) {
        self.id = id; self.name = name; self.modelRepo = modelRepo; self.datasetId = datasetId
        self.state = state; self.engine = engine; self.createdAt = createdAt; self.adapterPath = adapterPath
    }
}

struct RunStatus: Codable, Sendable, Equatable {
    let id: String
    let name: String
    let state: String
    let error: String?
    let numEvents: Int

    enum CodingKeys: String, CodingKey {
        case id, name, state, error
        case numEvents = "num_events"
    }
}

/// A single training-output event. Fields are optional because event types
/// (train / val / saved / info / status) carry different payloads.
struct TrainEvent: Codable, Sendable, Equatable {
    let event: String
    var iter: Int?
    var trainLoss: Double?
    var valLoss: Double?
    var lr: Double?
    var tokensPerSec: Double?
    var itPerSec: Double?
    var peakMemGb: Double?
    var trainedTokens: Int?
    var state: String?
    var error: String?
    var path: String?
    var text: String?

    enum CodingKeys: String, CodingKey {
        case event, iter, lr, state, error, path, text
        case trainLoss = "train_loss"
        case valLoss = "val_loss"
        case tokensPerSec = "tokens_per_sec"
        case itPerSec = "it_per_sec"
        case peakMemGb = "peak_mem_gb"
        case trainedTokens = "trained_tokens"
    }
}

/// A point in a training metric series.
struct MetricPoint: Identifiable, Sendable, Equatable {
    var id: Int { iter }
    let iter: Int
    let value: Double
}

/// Streams training events for a run. Abstracted so view models test with a fake.
protocol RunEventStreaming: Sendable {
    func stream(runId: String) -> AsyncStream<TrainEvent>
}
