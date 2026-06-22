import SwiftUI

struct PlaygroundView: View {
    @Bindable var model: PlaygroundModel

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            outputs
        }
        .navigationTitle("Playground")
        .task { await model.loadInputs() }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                if model.cachedModels.isEmpty {
                    Text("Download a model in the Hub tab first.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $model.modelRepo) {
                        Text("Select…").tag("")
                        ForEach(model.cachedModels) { Text($0.repoId).tag($0.repoId) }
                    }
                    Picker("Adapter", selection: $model.adapterRunId) {
                        Text("None (base only)").tag("")
                        ForEach(model.runs) { Text($0.name).tag($0.id) }
                    }
                }
            }
            TextEditor(text: $model.prompt)
                .font(.body)
                .frame(height: 64)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            HStack(spacing: 20) {
                slider("Temp", $model.temp, 0...1.5)
                slider("Top-p", $model.topP, 0...1)
                Stepper("Max tokens: \(model.maxTokens)", value: $model.maxTokens, in: 16...1024, step: 16)
                    .fixedSize()
                Spacer()
                Button { Task { await model.generate() } } label: {
                    if model.generating { ProgressView().controlSize(.small) } else {
                        Label("Generate", systemImage: "sparkles")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canGenerate)
            }
            if let error = model.error {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption)
            }
        }
        .padding(16)
    }

    private var outputs: some View {
        HStack(spacing: 0) {
            outputPane(title: model.hasAdapter ? "Base" : "Output", text: model.baseOutput)
            if model.hasAdapter {
                Divider()
                outputPane(title: "Finetuned", text: model.adapterOutput, accent: true)
            }
        }
    }

    private func outputPane(title: String, text: String, accent: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(accent ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                .padding(.horizontal, 16).padding(.top, 12)
            ScrollView {
                Text(text.isEmpty ? "—" : text)
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func slider(_ label: String, _ binding: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: 6) {
            Text(label).foregroundStyle(.secondary)
            Slider(value: binding, in: range).frame(width: 110)
            Text(binding.wrappedValue.formatted(.number.precision(.fractionLength(2))))
                .monospacedDigit().frame(width: 36)
        }
        .font(.caption)
    }
}
