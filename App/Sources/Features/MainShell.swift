import SwiftUI

/// The primary app shell shown once the backend is healthy.
struct MainShell: View {
    @State private var selection: AppSection
    @State private var overview: OverviewModel
    @State private var hub: HubBrowserModel
    @State private var datasets: DatasetBuilderModel
    @State private var training: TrainingModel
    @State private var playground: PlaygroundModel
    @State private var export: ExportModel
    @State private var settings: SettingsModel
    private let runtime: RuntimeInfo?

    init(
        api: any BackendAPI, progress: any ProgressStreaming,
        runEvents: any RunEventStreaming, inference: any InferenceStreaming,
        runtime: RuntimeInfo?
    ) {
        _selection = State(initialValue: Self.initialSection())
        _overview = State(initialValue: OverviewModel(api: api))
        _hub = State(initialValue: HubBrowserModel(api: api, progress: progress))
        _datasets = State(initialValue: DatasetBuilderModel(api: api))
        _training = State(initialValue: TrainingModel(api: api, events: runEvents))
        _playground = State(initialValue: PlaygroundModel(api: api, infer: inference))
        _export = State(initialValue: ExportModel(api: api))
        _settings = State(initialValue: SettingsModel(api: api))
        self.runtime = runtime
    }

    /// DEBUG-only: open on a given section via `open … --args --start-section <name>`,
    /// used to screenshot each page during visual review.
    private static func initialSection() -> AppSection {
        #if DEBUG
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--start-section"), i + 1 < args.count,
            let section = AppSection(rawValue: args[i + 1]) {
            return section
        }
        #endif
        return .home
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 210, ideal: 224, max: 280)
        } detail: {
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .underPageBackgroundColor).opacity(0.5))
        }
        .frame(minWidth: 960, minHeight: 640)
        .tint(Theme.accent)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            brand.padding(.horizontal, Theme.Space.s).padding(.top, Theme.Space.s)

            SidebarRow(section: .home, isSelected: selection == .home) { selection = .home }

            Eyebrow(text: "Workflow")
                .padding(.horizontal, Theme.Space.m).padding(.top, Theme.Space.m)
            ForEach(AppSection.workflow) { section in
                SidebarRow(section: section, isSelected: selection == section) { selection = section }
            }

            Spacer()

            SidebarRow(section: .settings, isSelected: selection == .settings) { selection = .settings }
        }
        .padding(Theme.Space.s)
    }

    private var brand: some View {
        HStack(spacing: Theme.Space.m) {
            Image("AppLogo")
                .resizable().interpolation(.high)
                .frame(width: 36, height: 36)
                .shadow(color: Theme.accent.opacity(0.3), radius: 4, y: 2)
            VStack(alignment: .leading, spacing: 0) {
                Text("TinyForge").font(Theme.rounded(17, .bold))
                Text("Local ML studio").font(.system(size: 10)).foregroundStyle(.secondary)
            }
        }
        .padding(.bottom, Theme.Space.xs)
    }

    @ViewBuilder
    private var detail: some View {
        switch selection {
        case .home: OverviewView(model: overview) { selection = $0 }
        case .hub: HubBrowserView(model: hub)
        case .datasets: DatasetBuilderView(model: datasets)
        case .train: TrainingView(model: training)
        case .playground: PlaygroundView(model: playground)
        case .export: ExportView(model: export)
        case .settings: SettingsView(model: settings, runtime: runtime)
        }
    }
}

private struct SidebarRow: View {
    let section: AppSection
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.m) {
                leading
                VStack(alignment: .leading, spacing: 1) {
                    Text(section.title).font(.system(size: 13, weight: .medium)).foregroundStyle(.primary)
                    Text(section.subtitle).font(.system(size: 10)).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, Theme.Space.s).padding(.vertical, 6)
            .background(
                isSelected ? Theme.accent.opacity(0.14) : .clear,
                in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var leading: some View {
        if let step = section.step {
            Text("\(step)")
                .font(Theme.rounded(11, .bold))
                .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(Theme.accent))
                .frame(width: 22, height: 22)
                .background(
                    isSelected ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(Theme.accent.opacity(0.15)),
                    in: Circle())
        } else {
            Image(systemName: section.icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isSelected ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(.secondary))
                .frame(width: 22, height: 22)
        }
    }
}

