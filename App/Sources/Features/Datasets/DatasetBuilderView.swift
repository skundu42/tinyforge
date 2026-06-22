import Charts
import SwiftUI

struct DatasetBuilderView: View {
    @Bindable var model: DatasetBuilderModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                sourceSection
                if model.preview != nil {
                    previewSection
                    formatSection
                    analysisSection
                    prepareSection
                } else if let error = model.previewError {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange).font(.callout).textSelection(.enabled)
                }
                registrySection
            }
            .padding(20)
            .frame(maxWidth: 840, alignment: .leading)
        }
        .navigationTitle("Dataset Builder")
        .task { await model.loadRegistry() }
    }

    private var sourceSection: some View {
        GroupBox("Source") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Source", selection: $model.sourceKind) {
                    Text("HuggingFace Hub").tag("hub")
                    Text("Local file").tag("local")
                }
                .pickerStyle(.segmented).labelsHidden()

                if model.sourceKind == "hub" {
                    TextField("Dataset repo — e.g. tatsu-lab/alpaca", text: $model.repoId)
                        .textFieldStyle(.roundedBorder)
                } else {
                    TextField("Path to .jsonl / .csv / .parquet", text: $model.localPath)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    TextField("Split", text: $model.split).textFieldStyle(.roundedBorder).frame(width: 130)
                    Spacer()
                    Button { Task { await model.loadPreview() } } label: {
                        if model.loadingPreview { ProgressView().controlSize(.small) } else { Text("Preview") }
                    }
                    .disabled(!model.canPreview || model.loadingPreview)
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(6)
        }
    }

    private var previewSection: some View {
        GroupBox("Preview · \(model.preview?.numRows ?? 0) rows") {
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow {
                        ForEach(model.columns, id: \.self) { Text($0).font(.caption.bold()) }
                    }
                    Divider()
                    ForEach(Array((model.preview?.rows ?? []).prefix(8).enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(model.columns, id: \.self) { column in
                                Text(row[column]?.display ?? "")
                                    .font(.caption).lineLimit(2)
                                    .frame(maxWidth: 240, alignment: .leading)
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(maxHeight: 220)
        }
    }

    private var formatSection: some View {
        GroupBox("Format → mlx-lm") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Target", selection: $model.spec.mode) {
                    Text("Text").tag("text")
                    Text("Prompt + Completion").tag("prompt_completion")
                    Text("Alpaca (instruction/input/output)").tag("alpaca")
                    Text("Chat messages").tag("messages")
                }
                switch model.spec.mode {
                case "text":
                    columnPicker("Text column", $model.spec.textColumn)
                case "prompt_completion":
                    columnPicker("Prompt column", $model.spec.promptColumn)
                    columnPicker("Completion column", $model.spec.completionColumn)
                case "alpaca":
                    columnPicker("Instruction", $model.spec.instructionColumn)
                    columnPicker("Input (optional)", $model.spec.inputColumn)
                    columnPicker("Output", $model.spec.outputColumn)
                case "messages":
                    columnPicker("Messages column", $model.spec.messagesColumn)
                default: EmptyView()
                }
            }
            .padding(6)
        }
    }

    private func columnPicker(_ label: String, _ binding: Binding<String>) -> some View {
        Picker(label, selection: binding) {
            ForEach(model.columns, id: \.self) { Text($0).tag($0) }
        }
    }

    private var analysisSection: some View {
        GroupBox("Token length (optional)") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Tokenizer repo — usually the base model", text: $model.tokenizerRepo)
                        .textFieldStyle(.roundedBorder)
                    Button { Task { await model.analyze() } } label: {
                        if model.analyzing { ProgressView().controlSize(.small) } else { Text("Analyze") }
                    }
                    .disabled(model.tokenizerRepo.isEmpty || model.analyzing)
                }
                if let stats = model.tokenStats {
                    HStack(spacing: 16) {
                        statLabel("min", stats.min)
                        statLabel("p50", stats.p50)
                        statLabel("mean", Int(stats.mean))
                        statLabel("p95", stats.p95)
                        statLabel("max", stats.max)
                    }
                    .font(.caption).monospacedDigit()
                    Chart(stats.histogram) { bin in
                        BarMark(x: .value("Tokens", bin.lo), y: .value("Count", bin.count))
                    }
                    .frame(height: 120)
                }
            }
            .padding(6)
        }
    }

    private func statLabel(_ name: String, _ value: Int) -> some View {
        VStack { Text("\(value)").bold(); Text(name).foregroundStyle(.secondary) }
    }

    private var prepareSection: some View {
        GroupBox("Prepare") {
            VStack(alignment: .leading, spacing: 10) {
                TextField("Dataset name", text: $model.datasetName).textFieldStyle(.roundedBorder)
                HStack {
                    Text("Validation split")
                    Slider(value: $model.valFraction, in: 0...0.5)
                    Text("\(Int(model.valFraction * 100))%").monospacedDigit().frame(width: 44)
                }
                HStack {
                    Button { Task { await model.prepare() } } label: {
                        if model.preparing { ProgressView().controlSize(.small) } else { Text("Prepare dataset") }
                    }
                    .disabled(!model.canPrepare || model.preparing)
                    .buttonStyle(.borderedProminent)
                    if let message = model.message {
                        Text(message).font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(6)
        }
    }

    private var registrySection: some View {
        GroupBox("Prepared datasets") {
            VStack(alignment: .leading, spacing: 6) {
                if model.registered.isEmpty {
                    Text("None yet — prepare one above.").font(.callout).foregroundStyle(.secondary)
                } else {
                    ForEach(model.registered) { dataset in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(dataset.name).font(.callout.weight(.medium))
                                Text("\(dataset.targetFormat) · \(dataset.trainRows) train / \(dataset.valRows) val")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button(role: .destructive) {
                                Task { await model.delete(dataset.id) }
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
            }
            .padding(6)
        }
    }
}
