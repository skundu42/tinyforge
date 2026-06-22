import Foundation
import Testing

@testable import TinyForge

struct BackendLauncherTests {
    @Test func buildsDevSpecFromEnvironment() throws {
        let env = [
            BackendLauncher.devPythonEnvKey: "/tmp/venv/bin/python",
            BackendLauncher.devBackendDirEnvKey: "/tmp/backend",
        ]

        let spec = try #require(BackendLauncher.resolveSpec(token: "abc", environment: env))

        #expect(spec.executable.path == "/tmp/venv/bin/python")
        #expect(spec.arguments == ["-m", "tinyforge", "--port", "0"])
        #expect(spec.workingDirectory?.path == "/tmp/backend")
        #expect(spec.token == "abc")
    }

    @Test func debugRepoSpecComputesBackendVenvFromSourcePath() {
        let fakeFile = "/work/macos-ml/App/Sources/Backend/BackendLauncher.swift"
        let expectedPython = "/work/macos-ml/backend/.venv/bin/python3"
        let fileManager = StubFileManager(executablePaths: [expectedPython])

        let spec = BackendLauncher.debugRepoSpec(
            token: "abc", file: fakeFile, fileManager: fileManager)

        #expect(spec?.executable.path == expectedPython)
        #expect(spec?.workingDirectory?.path == "/work/macos-ml/backend")
        #expect(spec?.arguments == ["-m", "tinyforge", "--port", "0"])
    }

    @Test func tokenIsUrlSafeHex() {
        // Must be safe in a WebSocket query string; base64's +,/,= would break it.
        let token = TokenGenerator.make()
        #expect(token.count == 64)
        #expect(token.allSatisfy { $0.isHexDigit })
    }

    @Test func debugRepoSpecReturnsNilWhenVenvMissing() {
        let fakeFile = "/work/macos-ml/App/Sources/Backend/BackendLauncher.swift"
        let spec = BackendLauncher.debugRepoSpec(
            token: "abc", file: fakeFile, fileManager: StubFileManager(executablePaths: []))
        #expect(spec == nil)
    }
}

private final class StubFileManager: FileManager, @unchecked Sendable {
    private let executablePaths: Set<String>
    init(executablePaths: Set<String>) {
        self.executablePaths = executablePaths
        super.init()
    }
    override func isExecutableFile(atPath path: String) -> Bool {
        executablePaths.contains(path)
    }
}
