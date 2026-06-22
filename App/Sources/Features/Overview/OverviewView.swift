import SwiftUI

struct OverviewView: View {
    @Bindable var model: OverviewModel
    let onSelect: (AppSection) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {
                hero
                Text("Your forge journey")
                    .font(Theme.rounded(17, .semibold))
                    .padding(.top, Theme.Space.s)
                VStack(spacing: 0) {
                    ForEach(Array(AppSection.workflow.enumerated()), id: \.element) { index, section in
                        stepRow(section, isLast: index == AppSection.workflow.count - 1)
                    }
                }
            }
            .padding(Theme.Space.xxl)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .navigationTitle("Home")
        .task { await model.load() }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            Eyebrow(text: "Welcome to TinyForge")
            Text("Forge a model on your Mac")
                .font(Theme.rounded(30, .bold))
            Text("Download a model, build a dataset, finetune it on your Apple Silicon GPU, try it out, and share it — all locally. New here? Just follow the steps below.")
                .font(.body).foregroundStyle(.secondary).frame(maxWidth: 600, alignment: .leading)
            HStack(spacing: Theme.Space.xxl) {
                Stat(value: "\(model.modelCount)", label: "models", tint: Theme.accent)
                Stat(value: "\(model.datasetCount)", label: "datasets", tint: Theme.accent)
                Stat(value: "\(model.finetuneCount)", label: "finetunes", tint: Theme.ember)
            }
            .padding(.top, Theme.Space.xs)
        }
        .padding(.bottom, Theme.Space.s)
    }

    @ViewBuilder
    private func stepRow(_ section: AppSection, isLast: Bool) -> some View {
        let state = state(for: section)
        HStack(alignment: .top, spacing: Theme.Space.m) {
            // Numbered spine: badge with a connector line down to the next step.
            VStack(spacing: 0) {
                StepBadge(number: section.step ?? 0, state: state)
                if !isLast {
                    Rectangle().fill(Theme.hairline)
                        .frame(width: 2).frame(maxHeight: .infinity)
                }
            }
            .frame(width: 30)

            Button { onSelect(section) } label: {
                stepCard(section, state: state)
            }
            .buttonStyle(.plain)
            .padding(.bottom, Theme.Space.m)
        }
    }

    private func stepCard(_ section: AppSection, state: StepState) -> some View {
        HStack(spacing: Theme.Space.m) {
            Image(systemName: section.icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(state == .upcoming ? AnyShapeStyle(.secondary) : AnyShapeStyle(Theme.accent))
                .frame(width: 34, height: 34)
                .background((state == .upcoming ? Color.secondary : Theme.accent).opacity(0.12),
                           in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: Theme.Space.s) {
                    Text(stepTitle(section)).font(.system(size: 15, weight: .semibold))
                    if state == .current { Pill(text: "Start here", tint: Theme.ember) }
                }
                Text(stepHint(section)).font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            statusView(section, state: state)
            Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
        }
        .card(Theme.Space.l)
        .overlay(
            RoundedRectangle(cornerRadius: Theme.radius, style: .continuous)
                .strokeBorder(state == .current ? Theme.ember.opacity(0.6) : .clear, lineWidth: 1.5))
    }

    @ViewBuilder
    private func statusView(_ section: AppSection, state: StepState) -> some View {
        if let count = model.count(for: section), count > 0 {
            Pill(text: "\(count) ready", tint: Theme.success)
        } else if state == .done {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.success)
        }
    }

    // Step copy
    private func stepTitle(_ s: AppSection) -> String {
        switch s {
        case .hub: "Get a model"
        case .datasets: "Build a dataset"
        case .train: "Finetune it"
        case .playground: "Try it out"
        case .export: "Save & share"
        default: s.title
        }
    }

    private func stepHint(_ s: AppSection) -> String {
        switch s {
        case .hub: "Download a small model from HuggingFace."
        case .datasets: "Turn your data into training examples."
        case .train: "Train a LoRA on the GPU and watch it learn."
        case .playground: "Chat with your finetuned model."
        case .export: "Export to safetensors, MLX, or GGUF — or push to the Hub."
        default: s.subtitle
        }
    }

    // Progress logic
    private func state(for section: AppSection) -> StepState {
        let done: [AppSection: Bool] = [
            .hub: model.modelCount > 0,
            .datasets: model.datasetCount > 0,
            .train: model.finetuneCount > 0,
            .playground: model.finetuneCount > 0,
            .export: model.finetuneCount > 0,
        ]
        if done[section] == true { return .done }
        // The current step is the first not-done step in workflow order.
        let firstPending = AppSection.workflow.first { done[$0] != true }
        return section == firstPending ? .current : .upcoming
    }
}

enum StepState { case done, current, upcoming }

struct StepBadge: View {
    let number: Int
    let state: StepState

    var body: some View {
        ZStack {
            Circle().fill(fill)
            Circle().strokeBorder(stroke, lineWidth: 1.5)
            Text("\(number)")
                .font(Theme.rounded(13, .bold))
                .foregroundStyle(state == .upcoming ? AnyShapeStyle(.secondary) : AnyShapeStyle(.white))
        }
        .frame(width: 30, height: 30)
        .background(Theme.surface, in: Circle())  // punch through the spine line
    }

    private var fill: AnyShapeStyle {
        switch state {
        case .done: AnyShapeStyle(Theme.sparkGradient)
        case .current: AnyShapeStyle(Theme.accent)
        case .upcoming: AnyShapeStyle(Color(nsColor: .controlBackgroundColor))
        }
    }
    private var stroke: Color {
        state == .upcoming ? Theme.hairline : .clear
    }
}
