import Foundation
import Observation

/// Drives the finetuning screen: choose model + dataset + hyperparameters, start
/// a run, ingest streamed events into live metric series, stop, and list runs.
@MainActor
@Observable
final class TrainingModel {
    // Config
    var name = ""
    var engine = "mlx"  // mlx (LLM LoRA) | torch (from-scratch MPS)
    var modelRepo = ""
    var datasetId = ""
    var fineTuneType = "lora"
    var iters = 100
    var numLayers = 16
    var batchSize = 1
    var learningRate = 1e-5
    var maxSeqLength = 512

    // Available inputs
    private(set) var datasets: [RegisteredDataset] = []
    private(set) var cachedModels: [CachedRepo] = []

    // Run state
    private(set) var activeRunId: String?
    private(set) var runState: String?
    private(set) var runError: String?
    private(set) var adapterPath: String?
    private(set) var runs: [RunRecord] = []

    // Live metrics
    private(set) var trainLoss: [MetricPoint] = []
    private(set) var valLoss: [MetricPoint] = []
    private(set) var throughput: [MetricPoint] = []
    private(set) var peakMem: [MetricPoint] = []
    private(set) var lastLR: Double?
    private(set) var lastIter: Int?

    private let api: any BackendAPI
    private let events: any RunEventStreaming

    init(api: any BackendAPI, events: any RunEventStreaming) {
        self.api = api
        self.events = events
    }

    var isRunning: Bool { runState == "running" }
    var isLLM: Bool { engine == "mlx" }
    var canStart: Bool {
        guard !name.isEmpty, !isRunning else { return false }
        return isLLM ? (!modelRepo.isEmpty && !datasetId.isEmpty) : true
    }

    func loadInputs() async {
        datasets = (try? await api.listDatasets()) ?? []
        cachedModels = ((try? await api.cacheInfo())?.repos ?? []).filter { $0.repoType == "model" }
        await loadRuns()
    }

    func loadRuns() async {
        runs = (try? await api.listRuns()) ?? []
    }

    func start() async {
        resetMetrics()
        let request = StartRunRequest(
            name: name, modelRepo: modelRepo, datasetId: datasetId, engine: engine,
            fineTuneType: fineTuneType, numLayers: numLayers, batchSize: batchSize, iters: iters,
            learningRate: learningRate, maxSeqLength: maxSeqLength
        )
        do {
            let record = try await api.startRun(request)
            activeRunId = record.id
            runState = "running"
            runError = nil
            await loadRuns()
            for await event in events.stream(runId: record.id) {
                ingest(event)
            }
        } catch {
            runState = "failed"
            runError = String(describing: error)
        }
        await loadRuns()
    }

    func ingest(_ event: TrainEvent) {
        switch event.event {
        case "train":
            if let iter = event.iter {
                lastIter = iter
                if let loss = event.trainLoss { trainLoss.append(MetricPoint(iter: iter, value: loss)) }
                if let tps = event.tokensPerSec { throughput.append(MetricPoint(iter: iter, value: tps)) }
                if let mem = event.peakMemGb { peakMem.append(MetricPoint(iter: iter, value: mem)) }
            }
            lastLR = event.lr ?? lastLR
        case "val":
            if let iter = event.iter, let loss = event.valLoss {
                valLoss.append(MetricPoint(iter: iter, value: loss))
            }
        case "saved":
            adapterPath = event.path
        case "status":
            runState = event.state
            runError = event.error
        default:
            break
        }
    }

    func stop() async {
        guard let id = activeRunId else { return }
        try? await api.stopRun(id: id)
        runState = "stopped"
    }

    private func resetMetrics() {
        trainLoss = []
        valLoss = []
        throughput = []
        peakMem = []
        lastLR = nil
        lastIter = nil
        adapterPath = nil
    }
}
