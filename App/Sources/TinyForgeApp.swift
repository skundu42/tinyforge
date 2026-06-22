import SwiftUI

enum AppEnvironment {
    /// True when the process is hosting an XCTest/Swift Testing bundle, so the
    /// app shouldn't auto-launch the backend during tests.
    static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}

@main
struct TinyForgeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appDelegate.backend)
                .task {
                    guard !AppEnvironment.isRunningTests else { return }
                    await appDelegate.backend.launchIfNeeded()
                }
        }
        .windowResizability(.contentSize)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let backend = BackendController()

    func applicationWillTerminate(_ notification: Notification) {
        backend.shutdownSync()
    }
}
