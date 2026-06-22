import SwiftUI

struct ContentView: View {
    @Environment(BackendController.self) private var backend

    var body: some View {
        switch backend.phase {
        case .healthy:
            if let api = backend.api, let progress = backend.progressClient,
                let runEvents = backend.runEventClient, let inference = backend.inferenceClient {
                MainShell(
                    api: api, progress: progress, runEvents: runEvents,
                    inference: inference, runtime: backend.runtime)
            } else {
                BackendStatusView()
            }
        case .idle, .launching, .unavailable:
            BackendStatusView()
                .frame(minWidth: 560, minHeight: 460)
        }
    }
}

/// Shown while the backend is starting or unavailable — branded and centered.
struct BackendStatusView: View {
    @Environment(BackendController.self) private var backend

    var body: some View {
        VStack(spacing: Theme.Space.xl) {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Theme.brandGradient)
                .frame(width: 88, height: 88)
                .overlay(
                    Image(systemName: "sparkle")
                        .font(.system(size: 40, weight: .bold))
                        .foregroundStyle(Theme.sparkGradient))
                .shadow(color: Theme.accent.opacity(0.4), radius: 16, y: 6)

            VStack(spacing: Theme.Space.s) {
                Text("TinyForge").font(Theme.rounded(28, .bold))
                Text("Train, finetune & experiment with tiny ML models")
                    .font(.callout).foregroundStyle(.secondary)
            }

            statusLine
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(Theme.Space.xxl)
    }

    @ViewBuilder
    private var statusLine: some View {
        switch backend.phase {
        case .idle:
            Label("Getting ready…", systemImage: "circle.dotted").foregroundStyle(.secondary)
        case .launching:
            HStack(spacing: Theme.Space.s) {
                ProgressView().controlSize(.small)
                Text("Starting the local engine…").foregroundStyle(.secondary)
            }
        case .unavailable(let message):
            VStack(spacing: Theme.Space.s) {
                Label("Couldn't start the engine", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(Theme.ember).font(.headline)
                Text(message)
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 420).textSelection(.enabled)
            }
            .card()
        case .healthy:
            Label("Ready", systemImage: "checkmark.seal.fill").foregroundStyle(Theme.success)
        }
    }
}
