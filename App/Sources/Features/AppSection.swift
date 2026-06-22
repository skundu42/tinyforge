import SwiftUI

/// The app's navigation destinations. The five workflow steps form one real
/// sequence — get a model → build data → finetune → test → export — so they
/// carry step numbers; Home and Settings don't.
enum AppSection: String, CaseIterable, Identifiable {
    case home, hub, datasets, train, playground, export, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .home: "Home"
        case .hub: "Models"
        case .datasets: "Datasets"
        case .train: "Finetune"
        case .playground: "Playground"
        case .export: "Export"
        case .settings: "Settings"
        }
    }

    var subtitle: String {
        switch self {
        case .home: "Overview"
        case .hub: "Browse & download"
        case .datasets: "Prepare training data"
        case .train: "Train on your Mac"
        case .playground: "Try your model"
        case .export: "Save & share"
        case .settings: "Account & cache"
        }
    }

    var icon: String {
        switch self {
        case .home: "house.fill"
        case .hub: "square.stack.3d.up.fill"
        case .datasets: "tablecells.fill"
        case .train: "bolt.fill"
        case .playground: "sparkles"
        case .export: "shippingbox.fill"
        case .settings: "gearshape.fill"
        }
    }

    /// Position in the finetuning workflow, or nil for Home/Settings.
    var step: Int? {
        switch self {
        case .hub: 1
        case .datasets: 2
        case .train: 3
        case .playground: 4
        case .export: 5
        default: nil
        }
    }

    static var workflow: [AppSection] { [.hub, .datasets, .train, .playground, .export] }
}
