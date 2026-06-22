import Foundation
import Observation

/// Drives the export screen: pick a completed run, choose a target format,
/// optionally push to the Hub, start the export, and poll until it finishes.
@MainActor
@Observable
final class ExportModel {
    var runId = ""
    var target = "safetensors"
    var qBits = 4
    var pushRepo = ""

    private(set) var runs: [RunRecord] = []
    private(set) var exports: [ExportStatus] = []
    private(set) var busy = false
    private(set) var message: String?

    private let api: any BackendAPI

    init(api: any BackendAPI) {
        self.api = api
    }

    var canExport: Bool { !runId.isEmpty && !busy }

    func loadInputs() async {
        runs = ((try? await api.listRuns()) ?? []).filter { $0.state == "completed" }
        await refresh()
    }

    func refresh() async {
        exports = (try? await api.listExports()) ?? []
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
                try? await Task.sleep(for: .seconds(1))
                status = try await api.getExport(id: status.id)
            }
            message = status.state == "completed"
                ? "Exported to \(status.outputPath ?? "")\(status.hubUrl.map { " · " + $0 } ?? "")"
                : "Export failed: \(status.error ?? "unknown")"
            await refresh()
        } catch {
            message = String(describing: error)
        }
    }
}
