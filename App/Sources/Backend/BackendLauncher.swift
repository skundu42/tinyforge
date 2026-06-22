import Foundation

/// Resolves how to launch the backend: a dev override via env vars, or the
/// Python runtime bundled inside the app at `Contents/Resources/python`.
enum BackendLauncher {
    static let devPythonEnvKey = "TINYFORGE_DEV_PYTHON"
    static let devBackendDirEnvKey = "TINYFORGE_DEV_BACKEND_DIR"

    static let arguments = ["-m", "tinyforge", "--port", "0"]

    static func resolveSpec(
        token: String,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main,
        fileManager: FileManager = .default
    ) -> LaunchSpec? {
        // 1) Development override via environment variables.
        if let python = environment[devPythonEnvKey], !python.isEmpty {
            let workingDirectory = environment[devBackendDirEnvKey]
                .flatMap { $0.isEmpty ? nil : URL(filePath: $0) }
            return LaunchSpec(
                executable: URL(filePath: python),
                arguments: arguments,
                workingDirectory: workingDirectory,
                token: token,
                extraEnvironment: [:]
            )
        }

        // 2) DEBUG convenience: self-locate the repo's dev venv so the app runs
        //    straight from Xcode without any env setup. Never compiled into release.
        #if DEBUG
        if let spec = debugRepoSpec(token: token, fileManager: fileManager) {
            return spec
        }
        #endif

        // 3) Python runtime bundled inside the app.
        if let resources = bundle.resourceURL {
            let python = resources.appending(path: "python/venv/bin/python3")
            if fileManager.isExecutableFile(atPath: python.path) {
                return LaunchSpec(
                    executable: python,
                    arguments: arguments,
                    workingDirectory: nil,
                    token: token,
                    extraEnvironment: [:]
                )
            }
        }

        return nil
    }

    #if DEBUG
    /// Derives the repo's dev backend (`backend/.venv/bin/python3`) from this
    /// source file's location, used only for running from Xcode in development.
    static func debugRepoSpec(
        token: String,
        file: String = #filePath,
        fileManager: FileManager = .default
    ) -> LaunchSpec? {
        let backendDir = URL(filePath: file)
            .deletingLastPathComponent()  // Backend
            .deletingLastPathComponent()  // Sources
            .deletingLastPathComponent()  // App
            .deletingLastPathComponent()  // repo root
            .appending(path: "backend")
        let python = backendDir.appending(path: ".venv/bin/python3")
        guard fileManager.isExecutableFile(atPath: python.path) else { return nil }
        return LaunchSpec(
            executable: python,
            arguments: arguments,
            workingDirectory: backendDir,
            token: token,
            extraEnvironment: [:]
        )
    }
    #endif
}

/// Generates the per-launch bearer token shared with the backend.
enum TokenGenerator {
    static func make() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        for index in bytes.indices {
            bytes[index] = UInt8.random(in: UInt8.min...UInt8.max)
        }
        return Data(bytes).base64EncodedString()
    }
}
