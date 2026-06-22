import SwiftUI

/// TinyForge design system — a cool indigo base with a warm forge-spark accent,
/// SF Rounded for friendly display type, and quiet, consistent surfaces.
enum Theme {
    // Brand
    static let accent = Color(red: 0.43, green: 0.34, blue: 0.81)   // Iris  #6E56CF
    static let ember = Color(red: 1.00, green: 0.54, blue: 0.24)    // Ember #FF8A3D
    static let glow = Color(red: 1.00, green: 0.76, blue: 0.29)     // Glow  #FFC24B
    static let success = Color(red: 0.18, green: 0.71, blue: 0.49)
    static let danger = Color(red: 0.90, green: 0.28, blue: 0.30)

    static let brandGradient = LinearGradient(
        colors: [accent, Color(red: 0.55, green: 0.36, blue: 0.86)],
        startPoint: .topLeading, endPoint: .bottomTrailing)
    static let sparkGradient = LinearGradient(
        colors: [glow, ember], startPoint: .top, endPoint: .bottom)

    // Surfaces (adaptive)
    static let surface = Color(nsColor: .controlBackgroundColor)
    static let hairline = Color(nsColor: .separatorColor)

    // Spacing & shape
    enum Space {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }
    static let radius: CGFloat = 12

    static func rounded(_ size: CGFloat, _ weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }
}

extension View {
    /// A quiet, elevated card surface with a hairline border.
    func card(_ padding: CGFloat = Theme.Space.l) -> some View {
        self
            .padding(padding)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.radius, style: .continuous)
                    .strokeBorder(Theme.hairline.opacity(0.7), lineWidth: 1))
            .shadow(color: .black.opacity(0.05), radius: 7, y: 2)
    }
}

// MARK: - Components

/// A small uppercase eyebrow label.
struct Eyebrow: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 11, weight: .semibold))
            .tracking(0.8)
            .foregroundStyle(Theme.accent)
    }
}

/// A consistent section header: icon chip + title + optional subtitle.
struct SectionTitle: View {
    let title: String
    var subtitle: String?
    var systemImage: String?

    var body: some View {
        HStack(spacing: Theme.Space.m) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                    .frame(width: 30, height: 30)
                    .background(Theme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.system(size: 16, weight: .semibold))
                if let subtitle {
                    Text(subtitle).font(.system(size: 12)).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// A titled card section — a branded header over content, on a card surface.
struct Panel<Content: View>: View {
    let title: String
    var subtitle: String?
    var systemImage: String?
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            SectionTitle(title: title, subtitle: subtitle, systemImage: systemImage)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}

/// A pill tag with a tinted background.
struct Pill: View {
    let text: String
    var tint: Color = Theme.accent
    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .medium))
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(tint.opacity(0.15), in: Capsule())
            .foregroundStyle(tint)
    }
}

/// A rounded numeric stat with a caption.
struct Stat: View {
    let value: String
    let label: String
    var tint: Color = .primary
    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(Theme.rounded(22, .bold)).foregroundStyle(tint)
            Text(label).font(.system(size: 11)).foregroundStyle(.secondary)
        }
    }
}

/// A friendly empty state: an invitation to act, not a dead end.
struct EmptyState: View {
    let systemImage: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: Theme.Space.m) {
            Image(systemName: systemImage)
                .font(.system(size: 34, weight: .regular))
                .foregroundStyle(Theme.accent.opacity(0.55))
            Text(title).font(.system(size: 16, weight: .semibold))
            Text(message)
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 360)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(ForgeButtonStyle())
                    .padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(Theme.Space.xxl)
    }
}

/// Primary action button — accent-filled, rounded.
struct ForgeButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(Theme.accent.opacity(configuration.isPressed ? 0.8 : 1.0),
                        in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .foregroundStyle(.white)
            .contentShape(Rectangle())
    }
}
