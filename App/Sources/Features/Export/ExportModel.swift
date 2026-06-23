import Foundation
import Observation

/// Drives the export screen: pick a completed run, choose a target format,
/// optionally push to the Hub, start the export, and poll until it finishes.
@MainActor
@Observable
final class ExportModel: LoadErrorReporting {
    var runId = ""
    var target = "safetensors"
    var qBits = 4
    var pushRepo = ""

    private(set) var runs: [RunRecord] = []
    private(set) var exports: [ExportStatus] = []
    private(set) var busy = false
    private(set) var message: String?
    var loadError: String?
    /// Interval between export status polls (overridable in tests).
    var pollInterval: Duration = .seconds(1)

    private let api: any BackendAPI
    private var exportTask: Task<Void, Never>?

    init(api: any BackendAPI) {
        self.api = api
    }

    var canExport: Bool { !runId.isEmpty && !busy }

    /// Runs the export+poll as a cancellable task so leaving the screen stops the
    /// polling loop instead of leaking it.
    func startExport() {
        exportTask?.cancel()
        busy = true
        exportTask = Task { [weak self] in await self?.start() }
    }

    func cancelPolling() {
        exportTask?.cancel()
        exportTask = nil
        busy = false
    }

    func loadInputs() async {
        runs = (await attempt("Load finetunes") { try await api.listRuns() } ?? [])
            .filter { $0.state == "completed" }
        await refresh()
    }

    func refresh() async {
        exports = await attempt("Load exports") { try await api.listExports() } ?? []
    }

    func start() async {
        busy = true
        message = nil
        defer { busy = false }
        do {
            let request = ExportRequest(
                runId: runId, target: target, qBits: qBits,
                pushRepo: pushRepo.isEmpty ? nil : pushRepo)
            var status = try await api.startExport(request)
            while status.state == "running" {
                do {
                    try await Task.sleep(for: pollInterval)
                } catch {
                    return  // cancelled (e.g. view dismissed) — stop polling
                }
                status = try await api.getExport(id: status.id)
            }
            message = status.state == "completed"
                ? "Exported to \(status.outputPath ?? "")\(status.hubUrl.map { " · " + $0 } ?? "")"
                : "Export failed: \(status.error ?? "unknown")"
            await refresh()
        } catch {
            message = error.localizedDescription
        }
    }
}
