import Darwin
import Foundation
import Observation

/// Orchestrates the backend lifecycle for the UI: generates a token, launches
/// the process, and probes health/runtime. Exposes observable state to SwiftUI.
@MainActor
@Observable
final class BackendController {
    enum Phase: Equatable {
        case idle
        case launching
        case healthy(HealthStatus)
        case unavailable(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var runtime: RuntimeInfo?
    private(set) var api: APIClient?
    private(set) var progressClient: DownloadProgressClient?
    private(set) var runEventClient: RunEventClient?
    private(set) var inferenceClient: InferenceClient?

    private let manager = BackendProcessManager()
    private let token = TokenGenerator.make()

    func launchIfNeeded() async {
        guard case .idle = phase else { return }
        await launch()
    }

    func launch() async {
        phase = .launching
        guard let spec = BackendLauncher.resolveSpec(token: token) else {
            phase = .unavailable(
                "No backend runtime found. For development, set \(BackendLauncher.devPythonEnvKey) "
                    + "to the backend venv's python and \(BackendLauncher.devBackendDirEnvKey) to the backend dir."
            )
            return
        }
        do {
            let port = try await manager.start(spec)
            let base = URL(string: "http://127.0.0.1:\(port)")!
            let client = APIClient(baseURL: base, token: token)
            self.api = client
            self.progressClient = DownloadProgressClient(baseURL: base, token: token)
            self.runEventClient = RunEventClient(baseURL: base, token: token)
            self.inferenceClient = InferenceClient(baseURL: base, token: token)
            let health = try await client.health()
            runtime = try? await client.runtime()
            phase = .healthy(health)
        } catch {
            phase = .unavailable(String(describing: error))
        }
    }

    func shutdown() async {
        await manager.stop()
    }

    /// Synchronously signal the backend to terminate. Safe to call from
    /// applicationWillTerminate; the Python parent-death watchdog is the backstop
    /// for force-quit/crash where this never runs.
    nonisolated func shutdownSync() {
        if let pid = manager.childPID {
            kill(pid, SIGTERM)
        }
    }
}
