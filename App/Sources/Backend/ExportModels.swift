import Foundation

struct ExportRequest: Codable, Sendable {
    var runId: String
    var target: String = "safetensors"
    var qBits: Int = 4
    var pushRepo: String?

    enum CodingKeys: String, CodingKey {
        case target
        case runId = "run_id"
        case qBits = "q_bits"
        case pushRepo = "push_repo"
    }
}

struct ExportStatus: Codable, Sendable, Identifiable, Equatable {
    let id: String
    let runId: String
    let target: String
    let state: String
    let error: String?
    let outputPath: String?
    let hubUrl: String?

    enum CodingKeys: String, CodingKey {
        case id, target, state, error
        case runId = "run_id"
        case outputPath = "output_path"
        case hubUrl = "hub_url"
    }
}
