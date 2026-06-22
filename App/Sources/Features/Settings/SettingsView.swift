import SwiftUI

struct SettingsView: View {
    @Bindable var model: SettingsModel
    var runtime: RuntimeInfo?

    var body: some View {
        Form {
            Section("HuggingFace Account") { authSection }
            Section("Backend") { backendSection }
            if let runtime, !runtime.engines.isEmpty {
                Section("Engines") { enginesSection(runtime) }
            }
            Section {
                cacheSection
            } header: {
                HStack {
                    Text("Local Model Cache")
                    Spacer()
                    if let cache = model.cache {
                        Text(ByteFormat.string(cache.sizeOnDisk))
                            .foregroundStyle(.secondary).monospacedDigit()
                    }
                }
            }
            if let message = model.message {
                Section { Text(message).font(.callout).foregroundStyle(.secondary) }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .task { await model.refresh() }
    }

    @ViewBuilder
    private var authSection: some View {
        if model.auth.loggedIn {
            HStack {
                Label("Signed in as \(model.auth.name ?? "?")", systemImage: "person.crop.circle.fill.badge.checkmark")
                    .foregroundStyle(.green)
                Spacer()
                Button("Sign out") { Task { await model.logout() } }
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                Text("Sign in with a HuggingFace token to access gated and private repositories.")
                    .font(.callout).foregroundStyle(.secondary)
                HStack {
                    SecureField("hf_…", text: $model.tokenInput)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await model.login() } }
                    Button("Sign in") { Task { await model.login() } }
                        .disabled(model.tokenInput.isEmpty || model.busy)
                }
                Link("Create a token →", destination: URL(string: "https://huggingface.co/settings/tokens")!)
                    .font(.caption)
            }
        }
    }

    @ViewBuilder
    private var cacheSection: some View {
        if let repos = model.cache?.repos, !repos.isEmpty {
            ForEach(repos) { repo in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(repo.repoId).font(.callout)
                        Text("\(repo.repoType) · \(repo.nbFiles) files")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(ByteFormat.string(repo.sizeOnDisk))
                        .font(.callout).foregroundStyle(.secondary).monospacedDigit()
                    Button(role: .destructive) {
                        Task { await model.delete(repo.repoId) }
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                }
            }
        } else {
            Text("No models cached yet. Download one from the Hub.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var backendSection: some View {
        LabeledContent("Status") {
            Label("Running", systemImage: "checkmark.circle.fill").foregroundStyle(Theme.success)
        }
        if let runtime {
            LabeledContent("Python", value: runtime.pythonVersion)
            LabeledContent("Platform", value: "\(runtime.platform) · \(runtime.machine)")
        }
    }

    @ViewBuilder
    private func enginesSection(_ runtime: RuntimeInfo) -> some View {
        ForEach(runtime.engines.sorted(by: { $0.key < $1.key }), id: \.key) { name, available in
            LabeledContent(engineLabel(name)) {
                Label(available ? "Available" : "Not installed",
                      systemImage: available ? "checkmark.circle.fill" : "minus.circle")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(available ? AnyShapeStyle(Theme.success) : AnyShapeStyle(.secondary))
            }
        }
    }

    private func engineLabel(_ key: String) -> String {
        switch key {
        case "mlx": "MLX"
        case "mlx_lm": "MLX-LM"
        case "mlx_vlm": "MLX-VLM"
        case "torch": "PyTorch"
        case "transformers": "Transformers"
        default: key
        }
    }
}
