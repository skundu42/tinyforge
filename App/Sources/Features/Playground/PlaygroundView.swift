import SwiftUI

struct PlaygroundView: View {
    @Bindable var model: PlaygroundModel

    var body: some View {
        VStack(spacing: 0) {
            promptPanel
                .padding(Theme.Space.l)
            Divider()
            outputs
        }
        .navigationTitle("Playground")
        .task { await model.loadInputs() }
        .onDisappear { model.stop() }
    }

    private var promptPanel: some View {
        Panel(title: "Prompt", subtitle: "Generate text and compare base vs. finetuned", systemImage: "text.bubble.fill") {
            VStack(alignment: .leading, spacing: Theme.Space.m) {
                if model.cachedModels.isEmpty && model.scratchRuns.isEmpty {
                    Text("Download a model in the Models tab first.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    HStack(spacing: Theme.Space.l) {
                        if !model.cachedModels.isEmpty {
                            Picker("Model", selection: $model.modelRepo) {
                                Text("Select…").tag("")
                                ForEach(model.cachedModels) { repo in
                                    Text(repo.pickerLabel).tag(repo.repoId)
                                }
                            }
                        }
                        if !model.scratchRuns.isEmpty {
                            Picker("From-scratch model", selection: Binding(
                                get: {
                                    model.scratchRuns.contains { $0.adapterPath == model.modelRepo }
                                        ? model.modelRepo : ""
                                },
                                set: { id in
                                    if let run = model.scratchRuns.first(where: { $0.adapterPath == id }) {
                                        model.selectScratchModel(run)
                                    }
                                })
                            ) {
                                Text("None").tag("")
                                ForEach(model.scratchRuns) { Text($0.name).tag($0.adapterPath) }
                            }
                        }
                        Picker("Adapter", selection: $model.adapterRunId) {
                            Text("None (base only)").tag("")
                            ForEach(model.runs) { Text($0.name).tag($0.id) }
                        }
                        .disabled(model.runNative)
                        Picker("Run on", selection: $model.runNative) {
                            Text("Backend").tag(false)
                            Text("Native (MLX)").tag(true)
                        }
                        .fixedSize()
                        .help("Native runs generation in Swift via MLX-Swift (base model only).")
                    }
                }

                if let selected = model.cachedModels.first(where: { $0.repoId == model.modelRepo }),
                   selected.isTooBigForSystem {
                    HStack { TooBigTag(); Spacer() }
                }

                TextEditor(text: $model.prompt)
                    .font(.body)
                    .frame(height: 70)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(Color(nsColor: .textBackgroundColor),
                               in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Theme.hairline))

                HStack(alignment: .bottom, spacing: Theme.Space.xl) {
                    samplingSlider("Temperature", value: $model.temp, range: 0...1.5)
                    samplingSlider("Top-p", value: $model.topP, range: 0...1)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Max tokens").font(.caption).foregroundStyle(.secondary)
                        Stepper("\(model.maxTokens)", value: $model.maxTokens, in: 16...1024, step: 16)
                            .fixedSize()
                    }
                    Spacer(minLength: Theme.Space.l)
                    if model.generating {
                        Button(role: .cancel) { model.stop() } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button { model.startGenerate() } label: {
                            Label("Generate", systemImage: "sparkles")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canGenerate)
                    }
                }

                if let error = model.error {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(Theme.ember).font(.caption)
                }
            }
        }
    }

    private func samplingSlider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(2))))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
        .frame(width: 160)
    }

    private var outputs: some View {
        HStack(spacing: 0) {
            outputPane(title: model.showsComparison ? "Base" : "Output", text: model.baseOutput, accent: false)
            if model.showsComparison {
                Divider()
                outputPane(title: "Finetuned", text: model.adapterOutput, accent: true)
            }
        }
    }

    private func outputPane(title: String, text: String, accent: Bool) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            HStack(spacing: 6) {
                if accent { Image(systemName: "sparkle").font(.caption2).foregroundStyle(Theme.ember) }
                Text(title).font(.caption.weight(.semibold))
                    .foregroundStyle(accent ? AnyShapeStyle(Theme.ember) : AnyShapeStyle(.secondary))
            }
            .padding(.horizontal, Theme.Space.l).padding(.top, Theme.Space.m)

            ScrollView {
                Text(text.isEmpty ? "Your model's response will appear here." : text)
                    .font(.body)
                    .foregroundStyle(text.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(Theme.Space.l)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
