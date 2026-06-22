import Foundation

/// Arbitrary JSON value, for dataset preview cells (which can be nested).
enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    var display: String {
        switch self {
        case .string(let value): return value
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .bool(let value): return String(value)
        case .null: return ""
        case .array, .object:
            guard let data = try? JSONEncoder().encode(self),
                let text = String(data: data, encoding: .utf8)
            else { return "…" }
            return text
        }
    }
}

struct DatasetSource: Codable, Sendable, Equatable {
    var kind: String = "hub"  // hub | local
    var repoId: String?
    var config: String?
    var path: String?
    var fileFormat: String?
    var split: String = "train"

    enum CodingKeys: String, CodingKey {
        case kind, config, path, split
        case repoId = "repo_id"
        case fileFormat = "file_format"
    }
}

struct DatasetPreview: Codable, Sendable {
    let columns: [String]
    let rows: [[String: JSONValue]]
    let numRows: Int

    enum CodingKeys: String, CodingKey {
        case columns, rows
        case numRows = "num_rows"
    }
}

struct FormatSpec: Codable, Sendable, Equatable {
    var mode: String = "text"  // text | prompt_completion | messages | alpaca
    var textColumn: String = "text"
    var promptColumn: String = "prompt"
    var completionColumn: String = "completion"
    var messagesColumn: String = "messages"
    var instructionColumn: String = "instruction"
    var inputColumn: String = "input"
    var outputColumn: String = "output"

    enum CodingKeys: String, CodingKey {
        case mode
        case textColumn = "text_column"
        case promptColumn = "prompt_column"
        case completionColumn = "completion_column"
        case messagesColumn = "messages_column"
        case instructionColumn = "instruction_column"
        case inputColumn = "input_column"
        case outputColumn = "output_column"
    }
}

struct HistogramBin: Codable, Sendable, Equatable, Identifiable {
    var id: String { "\(lo)-\(hi)" }
    let lo: Int
    let hi: Int
    let count: Int
}

struct TokenStats: Codable, Sendable, Equatable {
    let count: Int
    let min: Int
    let max: Int
    let mean: Double
    let p50: Int
    let p95: Int
    let histogram: [HistogramBin]
}

struct RegisteredDataset: Codable, Sendable, Identifiable, Equatable {
    let id: String
    let name: String
    let targetFormat: String
    let trainRows: Int
    let valRows: Int
    let createdAt: String
    let path: String

    enum CodingKeys: String, CodingKey {
        case id, name, path
        case targetFormat = "target_format"
        case trainRows = "train_rows"
        case valRows = "val_rows"
        case createdAt = "created_at"
    }
}
