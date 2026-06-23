import SwiftUI

struct HubBrowserView: View {
    @Bindable var model: HubBrowserModel
    @State private var selectedId: HubModel.ID?

    var body: some View {
        VSplitView {
            downloadedPane
            searchPane
        }
        .navigationTitle("Models")
        .task { await model.loadDownloaded() }
        .onDisappear { model.cancelStreaming() }
    }

    // Downloaded models are pinned to the top — always one glance away.
    private var downloadedPane: some View {
        ScrollView {
            Panel(
                title: "Downloaded models",
                subtitle: downloadedSubtitle,
                systemImage: "internaldrive.fill"
            ) {
                if model.downloadedModels.isEmpty {
                    Text("Nothing downloaded yet — search below to find a model. "
                        + "It'll appear here, ready to finetune.")
                        .font(.callout).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    VStack(spacing: 0) {
                        ForEach(Array(model.downloadedModels.enumerated()), id: \.element.id) { index, repo in
                            DownloadedModelRow(repo: repo) {
                                Task { await model.deleteDownloaded(repo.repoId) }
                            }
                            if index < model.downloadedModels.count - 1 { Divider() }
                        }
                    }
                }
            }
            .padding(Theme.Space.l)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minHeight: 150)
    }

    private var downloadedSubtitle: String {
        guard !model.downloadedModels.isEmpty else { return "Nothing on your Mac yet" }
        let count = model.downloadedModels.count
        return "\(count) on your Mac · \(ByteFormat.string(totalDownloadedBytes)) · ready to finetune"
    }

    private var totalDownloadedBytes: Int {
        model.downloadedModels.reduce(0) { $0 + $1.sizeOnDisk }
    }

    // The bottom half: search and browse the Hub.
    private var searchPane: some View {
        VStack(spacing: 0) {
            searchBar
            Divider()
            searchContent
        }
        .frame(minHeight: 260)
    }

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Search models — SmolLM, Llama 3.2, mlx-community…", text: $model.query)
                .textFieldStyle(.plain)
                .onSubmit { Task { await model.search() } }
            Picker("Sort", selection: $model.sort) {
                Text("Downloads").tag("downloads")
                Text("Likes").tag("likes")
                Text("Recent").tag("last_modified")
                Text("Trending").tag("trending_score")
            }
            .labelsHidden()
            .fixedSize()
            Button("Search") { Task { await model.search() } }
                .keyboardShortcut(.defaultAction)
        }
        .padding(12)
    }

    @ViewBuilder
    private var searchContent: some View {
        switch model.phase {
        case .idle:
            EmptyState(
                systemImage: "magnifyingglass",
                title: "Search for a model",
                message: "Try “SmolLM”, “Llama 3.2”, or “Qwen”. Models from the mlx-community account are ready to finetune.")
        case .searching:
            ProgressView("Searching…").frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message):
            EmptyState(
                systemImage: "exclamationmark.triangle",
                title: "Search didn't work",
                message: message)
        case .results(let models):
            if models.isEmpty {
                EmptyState(
                    systemImage: "magnifyingglass",
                    title: "No models found",
                    message: "Try a different search, or browse the mlx-community account.")
            } else {
                HStack(spacing: 0) {
                    List(models, selection: $selectedId) { item in
                        HubModelRow(model: item, progress: model.progress(for: item.id)).tag(item.id)
                    }
                    .listStyle(.inset)
                    .frame(minWidth: 360)
                    .onChange(of: selectedId) { _, id in
                        if let id { Task { await model.loadDetail(id) } }
                    }
                    Divider()
                    detailPane.frame(minWidth: 340, maxWidth: .infinity)
                }
            }
        }
    }

    @ViewBuilder
    private var detailPane: some View {
        if model.detailLoading {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let detail = model.selectedDetail {
            HubModelDetailView(detail: detail, progress: model.progress(for: detail.id)) {
                model.startDownload(detail.id)
            }
        } else {
            EmptyState(
                systemImage: "doc.text.magnifyingglass",
                title: "Pick a model",
                message: "Select a result on the left to see its files and download it.")
        }
    }
}

private struct DownloadedModelRow: View {
    let repo: CachedRepo
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: Theme.Space.m) {
            Image(systemName: "cube.fill").foregroundStyle(Theme.accent)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(repo.repoId).font(.callout.weight(.medium)).lineLimit(1)
                    if repo.isTooBigForSystem { TooBigTag() }
                }
                Text("\(repo.nbFiles) file\(repo.nbFiles == 1 ? "" : "s")")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Text(ByteFormat.string(repo.sizeOnDisk))
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .help("Remove from local cache")
        }
        .padding(.vertical, 6)
    }
}

private struct HubModelRow: View {
    let model: HubModel
    let progress: DownloadProgress?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(model.id).font(.body.weight(.medium)).lineLimit(1)
            HStack(spacing: 12) {
                if let library = model.libraryName { TagChip(text: library) }
                if let downloads = model.downloads {
                    countLabel("arrow.down.circle", downloads)
                }
                if let likes = model.likes { countLabel("heart", likes) }
                if model.gated {
                    Label("gated", systemImage: "lock").font(.caption2).foregroundStyle(.orange)
                }
            }
            if let progress, progress.state != "error" {
                ProgressView(value: progress.fraction).controlSize(.small)
            }
        }
        .padding(.vertical, 3)
    }

    private func countLabel(_ symbol: String, _ value: Int) -> some View {
        Label(value.formatted(.number.notation(.compactName)), systemImage: symbol)
            .font(.caption2).foregroundStyle(.secondary)
    }
}

private struct HubModelDetailView: View {
    let detail: HubModelDetail
    let progress: DownloadProgress?
    let onDownload: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text(detail.id).font(.title2.bold()).textSelection(.enabled)
                metadata
                downloadSection
                Divider()
                filesSection
                if let readme = detail.readme, !readme.isEmpty {
                    Divider()
                    Text("README").font(.headline)
                    Text(readme)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(16)
        }
    }

    private var metadata: some View {
        HStack(spacing: 12) {
            if let library = detail.libraryName { TagChip(text: library) }
            if let pipeline = detail.pipelineTag { TagChip(text: pipeline) }
            if let total = detail.totalSize {
                Label(ByteFormat.string(total), systemImage: "internaldrive")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if SystemMemory.isTooBig(sizeBytes: detail.totalSize) { TooBigTag() }
        }
    }

    @ViewBuilder
    private var downloadSection: some View {
        if let progress {
            switch progress.state {
            case "completed":
                Label("Downloaded to local cache", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            case "error":
                VStack(alignment: .leading, spacing: 4) {
                    Label("Download failed", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
                    if let error = progress.error {
                        Text(error).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    Button("Retry", action: onDownload)
                }
            default:
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: progress.fraction)
                    Text("\(ByteFormat.string(progress.downloadedBytes)) / \(ByteFormat.string(progress.totalBytes)) · \(Int(progress.fraction * 100))%")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                if SystemMemory.isTooBig(sizeBytes: detail.totalSize) { TooBigTag() }
                Button(action: onDownload) {
                    Label("Download", systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var filesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Files (\(detail.siblings.count))").font(.headline)
            ForEach(detail.siblings) { file in
                HStack {
                    Image(systemName: "doc").foregroundStyle(.secondary)
                    Text(file.filename).font(.callout).lineLimit(1)
                    Spacer()
                    if let size = file.size, size > 0 {
                        Text(ByteFormat.string(size)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
            }
        }
    }
}

private struct TagChip: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(.tint.opacity(0.15), in: .capsule)
            .foregroundStyle(.tint)
    }
}
