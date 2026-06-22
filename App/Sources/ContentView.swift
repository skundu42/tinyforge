import SwiftUI

struct ContentView: View {
    @Environment(BackendController.self) private var backend

    var body: some View {
        switch backend.phase {
        case .healthy:
            if let api = backend.api, let progress = backend.progressClient,
                let runEvents = backend.runEventClient {
                MainShell(api: api, progress: progress, runEvents: runEvents, runtime: backend.runtime)
            } else {
                BackendStatusView()
            }
        case .idle, .launching, .unavailable:
            BackendStatusView()
                .frame(minWidth: 460, minHeight: 360)
        }
    }
}

/// Shown while the backend is starting or unavailable.
struct BackendStatusView: View {
    @Environment(BackendController.self) private var backend

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                Image(systemName: "cpu.fill").font(.system(size: 28)).foregroundStyle(.tint)
                VStack(alignment: .leading) {
                    Text("TinyForge").font(.largeTitle.bold())
                    Text("Train, finetune & experiment with tiny ML models")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
            }
            Divider()
            switch backend.phase {
            case .idle:
                Label("Idle", systemImage: "circle").foregroundStyle(.secondary)
            case .launching:
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Starting backend…")
                }
            case .unavailable(let message):
                VStack(alignment: .leading, spacing: 6) {
                    Label("Backend unavailable", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.headline)
                    Text(message).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
            case .healthy:
                Label("Backend healthy", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
            }
            Spacer()
        }
        .padding(28)
    }
}
