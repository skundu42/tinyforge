import SwiftUI

struct ExportView: View {
    @Bindable var model: ExportModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Panel(title: "Export a finetune", subtitle: "Turn a run into a standalone model", systemImage: "shippingbox.fill") {
                    VStack(alignment: .leading, spacing: 10) {
                        if model.runs.isEmpty {
                            Text("Complete a finetune in the Finetune tab first.")
                                .font(.callout).foregroundStyle(.secondary)
                        } else {
                            Picker("Run", selection: $model.runId) {
                                Text("Select…").tag("")
                                ForEach(model.runs) { Text($0.name).tag($0.id) }
                            }
                        }

                        Picker("Format", selection: $model.target) {
                            Text("safetensors").tag("safetensors")
                            Text("MLX").tag("mlx")
                            Text("GGUF").tag("gguf")
                            Text("Core ML").tag("coreml")
                        }
                        .pickerStyle(.segmented)
                        if model.target == "coreml" {
                            Text("Core ML works with from-scratch and vision runs (not LLM/MLX adapters).")
                                .font(.caption).foregroundStyle(.secondary)
                        }

                        if model.target == "mlx" {
                            Stepper("Quantization: \(model.qBits)-bit", value: $model.qBits, in: 2...8)
                                .fixedSize()
                        }

                        HStack {
                            TextField("Push to Hub repo (optional) — e.g. you/model", text: $model.pushRepo)
                                .textFieldStyle(.roundedBorder)
                        }

                        HStack {
                            Button { Task { await model.start() } } label: {
                                if model.busy { ProgressView().controlSize(.small) } else {
                                    Label("Export", systemImage: "shippingbox.fill")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!model.canExport)
                            if let message = model.message {
                                Text(message).font(.caption).foregroundStyle(.secondary)
                                    .textSelection(.enabled).lineLimit(2)
                            }
                        }
                    }
                }

                Panel(title: "Exports", systemImage: "clock.arrow.circlepath") {
                    VStack(alignment: .leading, spacing: 6) {
                        if model.exports.isEmpty {
                            Text("No exports yet.").font(.callout).foregroundStyle(.secondary)
                        } else {
                            ForEach(model.exports) { export in
                                HStack {
                                    Image(systemName: icon(export.state)).foregroundStyle(color(export.state))
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(export.target).font(.callout.weight(.medium))
                                        if let path = export.outputPath {
                                            Text(path).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                        }
                                    }
                                    Spacer()
                                    if let url = export.hubUrl, let link = URL(string: url) {
                                        Link("Hub", destination: link).font(.caption)
                                    }
                                    Text(export.state).font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .navigationTitle("Export")
        .task { await model.loadInputs() }
    }

    private func icon(_ state: String) -> String {
        switch state {
        case "completed": "checkmark.circle.fill"
        case "failed": "xmark.octagon.fill"
        default: "circle.dotted"
        }
    }

    private func color(_ state: String) -> Color {
        switch state {
        case "completed": .green
        case "failed": .red
        default: .secondary
        }
    }
}
