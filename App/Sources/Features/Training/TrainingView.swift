import Charts
import SwiftUI

struct TrainingView: View {
    @Bindable var model: TrainingModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                configSection
                statusBar
                if !model.trainLoss.isEmpty || model.isRunning {
                    dashboards
                }
                runHistory
            }
            .padding(20)
            .frame(maxWidth: 920, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .navigationTitle("Finetune")
        .task { await model.loadInputs() }
    }

    private var configSection: some View {
        Panel(title: "New finetune", subtitle: "Configure and start a run", systemImage: "bolt.fill") {
            VStack(alignment: .leading, spacing: 10) {
                TextField("Run name", text: $model.name).textFieldStyle(.roundedBorder)

                Picker("Engine", selection: $model.engine) {
                    Text("LLM LoRA (MLX)").tag("mlx")
                    Text("Tiny LM from scratch").tag("lm")
                }
                .pickerStyle(.segmented)

                if model.isLLM {
                    if model.cachedModels.isEmpty {
                        hint("Download a model in the Hub tab first.")
                    } else {
                        Picker("Base model", selection: $model.modelRepo) {
                            Text("Select…").tag("")
                            ForEach(model.cachedModels) { Text($0.repoId).tag($0.repoId) }
                        }
                    }

                    if model.datasets.isEmpty {
                        hint("Prepare a dataset in the Datasets tab first.")
                    } else {
                        Picker("Dataset", selection: $model.datasetId) {
                            Text("Select…").tag("")
                            ForEach(model.datasets) { Text($0.name).tag($0.id) }
                        }
                    }

                    Picker("Method", selection: $model.fineTuneType) {
                        Text("LoRA").tag("lora")
                        Text("DoRA").tag("dora")
                        Text("Full").tag("full")
                    }
                    .pickerStyle(.segmented)
                } else {
                    hint("Trains a small Llama-style LM from scratch on your dataset (Apple Silicon GPU). Pick a size; advanced knobs override it.")
                    if model.datasets.isEmpty {
                        hint("Prepare a dataset in the Datasets tab first.")
                    } else {
                        Picker("Dataset", selection: $model.datasetId) {
                            Text("Select…").tag("")
                            ForEach(model.datasets) { Text($0.name).tag($0.id) }
                        }
                    }
                    Picker("Model size", selection: $model.modelSize) {
                        Text("Tiny (~1–3M)").tag("tiny")
                        Text("Small (~8–15M)").tag("small")
                        Text("Medium (~30–60M)").tag("medium")
                        Text("Advanced").tag("custom")
                    }
                    .pickerStyle(.segmented)
                    if model.modelSize == "custom" {
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                            GridRow {
                                stepper("Hidden size", $model.hiddenSize, 64...1024, step: 64)
                                stepper("Layers", $model.numLayers, 1...24)
                            }
                            GridRow {
                                stepper("Heads", $model.numHeads, 1...16)
                                stepper("Context", $model.contextLength, 64...2048, step: 64)
                            }
                            GridRow {
                                stepper("Vocab size", $model.vocabSize, 1000...50000, step: 1000)
                            }
                        }
                    }
                }

                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                    GridRow {
                        stepper("Iterations", $model.iters, 1...5000, step: 10)
                        stepper("LoRA layers", $model.numLayers, 1...48)
                    }
                    GridRow {
                        stepper("Batch size", $model.batchSize, 1...16)
                        stepper("Max seq length", $model.maxSeqLength, 64...4096, step: 64)
                    }
                }

                HStack {
                    Text("Learning rate").frame(width: 110, alignment: .leading)
                    TextField("1e-5", value: $model.learningRate, format: .number.notation(.scientific))
                        .textFieldStyle(.roundedBorder).frame(width: 120)
                }

                HStack {
                    if model.isRunning {
                        Button(role: .destructive) { Task { await model.stop() } } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                    } else {
                        Button { Task { await model.start() } } label: {
                            Label("Start finetune", systemImage: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canStart)
                    }
                }
            }
        }
    }

    private var statusBar: some View {
        TimelineView(.periodic(from: Date(), by: 2)) { _ in
            let snapshot = SystemTelemetry.sample()
            HStack(spacing: 18) {
                if let state = model.runState {
                    Label(state.capitalized, systemImage: stateIcon(state))
                        .foregroundStyle(stateColor(state))
                }
                if let iter = model.lastIter { metric("iter", "\(iter)") }
                if let lr = model.lastLR { metric("lr", lr.formatted(.number.notation(.scientific))) }
                metric("GPU budget", String(format: "%.1f GB", snapshot.gpuBudgetGB))
                metric("thermal", snapshot.thermal)
                Spacer()
                if let path = model.adapterPath {
                    Label("Adapter saved", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green).help(path)
                }
            }
            .font(.callout)
        }
    }

    private var dashboards: some View {
        VStack(spacing: 14) {
            Panel(title: "Loss", subtitle: "Lower is better — train vs. validation", systemImage: "chart.line.downtrend.xyaxis") {
                Chart {
                    ForEach(model.trainLoss) { point in
                        LineMark(x: .value("Iter", point.iter), y: .value("Train", point.value))
                            .foregroundStyle(by: .value("Series", "train"))
                    }
                    ForEach(model.valLoss) { point in
                        LineMark(x: .value("Iter", point.iter), y: .value("Val", point.value))
                            .foregroundStyle(by: .value("Series", "val"))
                    }
                }
                .frame(height: 200)
            }
            HStack(spacing: 14) {
                Panel(title: "Throughput", subtitle: "tokens / sec", systemImage: "speedometer") {
                    Chart(model.throughput) { point in
                        LineMark(x: .value("Iter", point.iter), y: .value("tok/s", point.value))
                            .foregroundStyle(.orange)
                    }
                    .frame(height: 150)
                }
                Panel(title: "Peak memory", subtitle: "GB", systemImage: "memorychip.fill") {
                    Chart(model.peakMem) { point in
                        AreaMark(x: .value("Iter", point.iter), y: .value("GB", point.value))
                            .foregroundStyle(.purple.opacity(0.5))
                    }
                    .frame(height: 150)
                }
            }
        }
    }

    private var runHistory: some View {
        Panel(title: "Run history", systemImage: "clock.arrow.circlepath") {
            VStack(alignment: .leading, spacing: 6) {
                if model.runs.isEmpty {
                    hint("No runs yet.")
                } else {
                    ForEach(model.runs) { run in
                        HStack {
                            Image(systemName: stateIcon(run.state)).foregroundStyle(stateColor(run.state))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(run.name).font(.callout.weight(.medium))
                                Text(run.modelRepo).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer()
                            Text(run.state).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    // Helpers

    private func hint(_ text: String) -> some View {
        Text(text).font(.callout).foregroundStyle(.secondary)
    }

    private func metric(_ name: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(value).bold().monospacedDigit()
            Text(name).foregroundStyle(.secondary)
        }
    }

    private func stepper(_ label: String, _ binding: Binding<Int>, _ range: ClosedRange<Int>, step: Int = 1) -> some View {
        HStack {
            Stepper(value: binding, in: range, step: step) {
                HStack { Text(label); Spacer(); Text("\(binding.wrappedValue)").monospacedDigit().foregroundStyle(.secondary) }
            }
        }
        .frame(minWidth: 230)
    }

    private func stateIcon(_ state: String) -> String {
        switch state {
        case "running": "circle.dotted"
        case "completed": "checkmark.circle.fill"
        case "failed": "xmark.octagon.fill"
        case "stopped": "stop.circle.fill"
        default: "circle"
        }
    }

    private func stateColor(_ state: String) -> Color {
        switch state {
        case "completed": .green
        case "failed": .red
        case "running": .blue
        default: .secondary
        }
    }
}
