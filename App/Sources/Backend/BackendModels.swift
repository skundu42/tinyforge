import Foundation

/// The one-line readiness announcement the Python backend prints on stdout.
struct ReadyLine: Codable, Equatable, Sendable {
    let event: String
    let port: Int
    let token: String?
}

/// `GET /health` response (public liveness probe).
struct HealthStatus: Codable, Equatable, Sendable {
    let status: String
    let name: String
    let version: String
}

/// `GET /v1/runtime` response (engine availability + interpreter info).
struct RuntimeInfo: Codable, Equatable, Sendable {
    let pythonVersion: String
    let platform: String
    let machine: String
    let engines: [String: Bool]

    enum CodingKeys: String, CodingKey {
        case pythonVersion = "python_version"
        case platform
        case machine
        case engines
    }
}

/// Everything needed to launch the backend process.
struct LaunchSpec: Sendable, Equatable {
    var executable: URL
    var arguments: [String]
    var workingDirectory: URL?
    var token: String
    var extraEnvironment: [String: String]
}
