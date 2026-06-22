import Foundation
import Testing

@testable import TinyForge

/// Offline end-to-end for M2: spawns the real backend with a temp data dir,
/// builds a local JSONL dataset, then previews + prepares it and verifies the
/// registered train split is in mlx-lm "completion" format. Hermetic (no network).
struct DatasetIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let devPython = repoRoot.appending(path: "backend/.venv/bin/python3")
    static let backendDir = repoRoot.appending(path: "backend")

    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: devPython.path)))
    func prepareLocalAlpacaDatasetEndToEnd() async throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appending(path: "tf-ds-\(UUID().uuidString)")
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }

        let dataDir = tmp.appending(path: "appdata")
        let dataFile = tmp.appending(path: "train.jsonl")
        let rows = [
            #"{"instruction":"Add","input":"1+1","output":"2"}"#,
            #"{"instruction":"Capital","input":"France","output":"Paris"}"#,
            #"{"instruction":"Upper","input":"hi","output":"HI"}"#,
            #"{"instruction":"Echo","input":"yo","output":"yo"}"#,
        ].joined(separator: "\n")
        try rows.write(to: dataFile, atomically: true, encoding: .utf8)

        let token = TokenGenerator.make()
        let spec = LaunchSpec(
            executable: Self.devPython, arguments: BackendLauncher.arguments,
            workingDirectory: Self.backendDir, token: token,
            extraEnvironment: [
                "TINYFORGE_DATA_DIR": dataDir.path,
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            ]
        )
        let manager = BackendProcessManager()
        let port = try await manager.start(spec, readyTimeout: .seconds(60))
        let api = APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!, token: token)

        var caught: Error?
        do {
            let source = DatasetSource(kind: "local", path: dataFile.path, fileFormat: "json", split: "train")
            let preview = try await api.previewDataset(source, limit: 10)
            #expect(preview.columns.contains("instruction"))
            #expect(preview.numRows == 4)

            var format = FormatSpec()
            format.mode = "alpaca"
            let record = try await api.prepareDataset(
                name: "math-e2e", source: source, spec: format,
                valFraction: 0.25, seed: 0, maxRows: nil)

            #expect(record.targetFormat == "completion")
            #expect(record.trainRows + record.valRows == 4)

            let trainFile = URL(filePath: record.path).appending(path: "train.jsonl")
            let content = try String(contentsOf: trainFile, encoding: .utf8)
            #expect(content.contains("\"prompt\""))
            #expect(content.contains("\"completion\""))
        } catch {
            caught = error
        }

        await manager.stop()
        if let caught { throw caught }
    }
}
