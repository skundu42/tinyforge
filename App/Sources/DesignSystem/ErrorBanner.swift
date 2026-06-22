import SwiftUI

/// A dismissible inline error banner for surfacing load/refresh failures that
/// would otherwise be swallowed. Driven by a view model's `loadError`
/// (see `LoadErrorReporting`).
struct ErrorBanner: View {
    let message: String
    var onDismiss: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.danger)
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.primary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: Theme.Space.s)
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, Theme.Space.m)
        .padding(.vertical, Theme.Space.s)
        .background(Theme.danger.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
            .strokeBorder(Theme.danger.opacity(0.35), lineWidth: 1))
    }
}

extension View {
    /// Pins a dismissible `ErrorBanner` above the content whenever `message` is
    /// non-nil. Used to surface `LoadErrorReporting.loadError`.
    func loadErrorBanner(_ message: String?, onDismiss: @escaping () -> Void) -> some View {
        safeAreaInset(edge: .top) {
            if let message {
                ErrorBanner(message: message, onDismiss: onDismiss)
                    .padding(.horizontal, Theme.Space.xl)
                    .padding(.top, Theme.Space.m)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: message)
    }
}
