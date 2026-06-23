import SwiftUI

/// Warning capsule shown on a model whose size exceeds ~80% of the machine's RAM.
/// The decision (`SystemMemory.isTooBig`) lives in the model layer; this is purely
/// the presentation.
struct TooBigTag: View {
    var body: some View {
        Label("Too big for your system", systemImage: "exclamationmark.triangle.fill")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Theme.ember.opacity(0.18), in: .capsule)
            .foregroundStyle(Theme.ember)
            .help(
                "This model is larger than 80% of your \(SystemMemory.physicalMemoryString) "
                + "of RAM — it may fail to load or run very slowly.")
    }
}
