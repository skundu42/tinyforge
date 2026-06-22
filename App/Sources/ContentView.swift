import SwiftUI

struct ContentView: View {
    @Environment(BackendController.self) private var backend

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            header
            Divider()
            statusSection
            if let runtime = backend.runtime {
                Divider()
                RuntimeView(runtime: runtime)
            }
            Spacer()
        }
        .padding(28)
        .frame(minWidth: 460, minHeight: 360)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 28))
                .foregroundStyle(.tint)
            VStack(alignment: .leading) {
                Text("TinyForge").font(.largeTitle.bold())
                Text("Train, finetune & experiment with tiny ML models")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch backend.phase {
        case .idle:
            Label("Idle", systemImage: "circle")
                .foregroundStyle(.secondary)
        case .launching:
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Starting backend…")
            }
        case .healthy(let health):
            VStack(alignment: .leading, spacing: 6) {
                Label("Backend healthy", systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .font(.headline)
                Text("\(health.name) · v\(health.version)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        case .unavailable(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label("Backend unavailable", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.headline)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }
}

private struct RuntimeView: View {
    let runtime: RuntimeInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Runtime").font(.headline)
            Text("Python \(runtime.pythonVersion) · \(runtime.platform)/\(runtime.machine)")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Engines").font(.subheadline.bold()).padding(.top, 4)
            ForEach(runtime.engines.sorted(by: { $0.key < $1.key }), id: \.key) { name, available in
                Label(name, systemImage: available ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(available ? .green : .secondary)
                    .font(.callout)
            }
        }
    }
}
