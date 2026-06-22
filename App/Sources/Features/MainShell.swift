import SwiftUI

/// The primary app shell shown once the backend is healthy.
struct MainShell: View {
    enum Section: String, CaseIterable, Identifiable {
        case hub = "Hub"
        case datasets = "Datasets"
        case train = "Finetune"
        case settings = "Settings"
        var id: String { rawValue }
        var icon: String {
            switch self {
            case .hub: "square.stack.3d.up.fill"
            case .datasets: "tablecells.fill"
            case .train: "bolt.fill"
            case .settings: "gearshape.fill"
            }
        }
    }

    @State private var selection: Section = .hub
    @State private var hub: HubBrowserModel
    @State private var datasets: DatasetBuilderModel
    @State private var training: TrainingModel
    @State private var settings: SettingsModel
    private let runtime: RuntimeInfo?

    init(
        api: any BackendAPI, progress: any ProgressStreaming,
        runEvents: any RunEventStreaming, runtime: RuntimeInfo?
    ) {
        _hub = State(initialValue: HubBrowserModel(api: api, progress: progress))
        _datasets = State(initialValue: DatasetBuilderModel(api: api))
        _training = State(initialValue: TrainingModel(api: api, events: runEvents))
        _settings = State(initialValue: SettingsModel(api: api))
        self.runtime = runtime
    }

    var body: some View {
        NavigationSplitView {
            List(Section.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon).tag(section)
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190, max: 240)
            .safeAreaInset(edge: .bottom) { BackendBadge(runtime: runtime) }
        } detail: {
            switch selection {
            case .hub: HubBrowserView(model: hub)
            case .datasets: DatasetBuilderView(model: datasets)
            case .train: TrainingView(model: training)
            case .settings: SettingsView(model: settings)
            }
        }
        .navigationTitle("TinyForge")
        .frame(minWidth: 920, minHeight: 600)
    }
}

private struct BackendBadge: View {
    let runtime: RuntimeInfo?

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(.green).frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 1) {
                Text("Backend healthy").font(.caption.weight(.medium))
                if let runtime {
                    Text("Python \(runtime.pythonVersion) · \(runtime.machine)")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.4), in: .rect(cornerRadius: 8))
        .padding(8)
    }
}
