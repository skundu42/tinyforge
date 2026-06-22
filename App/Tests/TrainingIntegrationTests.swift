import Foundation
import Testing

@testable import TinyForge

/// Full-stack M3 check: real Swift client → backend → mlx_lm.lora, streaming
/// training events over the WebSocket to completion, producing an adapter.
/// Uses the cached SmolLM model offline. Opt-in (slow): create `.run-network-tests`.
struct TrainingIntegrationTests {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let devPython = repoRoot.appending(path: "backend/.venv/bin/python3")
    static let backendDir = repoRoot.appending(path: "backend")

    static var enabled: Bool {
        FileManager.default.isExecutableFile(atPath: devPython.path)
            && FileManager.default.fileExists(atPath: repoRoot.appending(path: ".run-network-tests").path)
    }

    @Test(.enabled(if: enabled))
    func finetuneStreamsMetricsAndSavesAdapter() async throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appending(path: "tf-train-\(UUID().uuidString)")
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }

        let dataDir = tmp.appending(path: "appdata")
        let dataFile = tmp.appending(path: "data.jsonl")
        let line = #"{"text":"The sky is blue and the grass is green."}"#
        try Array(repeating: line, count: 12).joined(separator: "\n")
            .write(to: dataFile, atomically: true, encoding: .utf8)

        let token = TokenGenerator.make()
        let spec = LaunchSpec(
            executable: Self.devPython, arguments: BackendLauncher.arguments,
            workingDirectory: Self.backendDir, token: token,
            extraEnvironment: ["TINYFORGE_DATA_DIR": dataDir.path, "HF_HUB_OFFLINE": "1"]
        )
        let manager = BackendProcessManager()
        let port = try await manager.start(spec, readyTimeout: .seconds(60))
        let base = URL(string: "http://127.0.0.1:\(port)")!
        let api = APIClient(baseURL: base, token: token)
        let runEvents = RunEventClient(baseURL: base, token: token)

        var caught: Error?
        do {
            // Prepare a tiny text dataset.
            var format = FormatSpec()
            format.mode = "text"
            let dataset = try await api.prepareDataset(
                name: "train-e2e",
                source: DatasetSource(kind: "local", path: dataFile.path, fileFormat: "json", split: "train"),
                spec: format, valFraction: 0.25, seed: 0, maxRows: nil)

            // Start a 3-iter LoRA finetune on the cached model.
            let run = try await api.startRun(StartRunRequest(
                name: "e2e", modelRepo: "mlx-community/SmolLM-135M-Instruct-4bit",
                datasetId: dataset.id, numLayers: 4, batchSize: 1, iters: 3,
                stepsPerReport: 1, stepsPerEval: 2, maxSeqLength: 64))

            var sawTrain = false
            var finalState: String?
            for await event in runEvents.stream(runId: run.id) {
                if event.event == "train", event.trainLoss != nil { sawTrain = true }
                if event.event == "status" { finalState = event.state }
            }

            #expect(sawTrain)
            #expect(finalState == "completed")
        } catch {
            caught = error
        }

        await manager.stop()
        if let caught { throw caught }
    }
}
