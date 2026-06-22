import Foundation

/// Wraps a non-Sendable value so it can cross a concurrency boundary. Used for
/// FileHandle, which is safe to read from a single consuming task.
final class UncheckedBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// Thread-safe holder for the child PID so it can be signalled from a
/// synchronous context (e.g. applicationWillTerminate) outside the actor.
final class PIDHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var value: pid_t?
    var pid: pid_t? {
        get { lock.withLock { value } }
        set { lock.withLock { value = newValue } }
    }
}

/// Accumulates the backend's stderr (capped) so failures can be reported.
actor StderrCollector {
    private var lines: [String] = []
    private let cap = 200

    func append(_ line: String) {
        lines.append(line)
        if lines.count > cap {
            lines.removeFirst(lines.count - cap)
        }
    }

    func text() -> String { lines.joined(separator: "\n") }
}

/// Spawns and supervises the Python backend process.
///
/// `start` launches the process, drains stderr to avoid pipe deadlock, and
/// resolves once the JSON ready line is seen on stdout (or times out / the
/// process exits first).
actor BackendProcessManager {
    enum BackendError: Error, Equatable {
        case alreadyRunning
        case exitedBeforeReady
        case readyTimeout
    }

    enum State: Equatable, Sendable {
        case idle
        case starting
        case running(port: Int)
        case stopped(code: Int32)
        case failed(String)
    }

    private(set) var state: State = .idle
    private var process: Process?
    private var stderrTask: Task<Void, Never>?
    private let pidHolder = PIDHolder()

    /// The child PID, readable from any context for emergency termination.
    nonisolated var childPID: pid_t? { pidHolder.pid }

    func start(_ spec: LaunchSpec, readyTimeout: Duration = .seconds(30)) async throws -> Int {
        guard process == nil else { throw BackendError.alreadyRunning }

        let proc = Process()
        proc.executableURL = spec.executable
        proc.arguments = spec.arguments
        proc.currentDirectoryURL = spec.workingDirectory

        var env = ProcessInfo.processInfo.environment
        env["TINYFORGE_TOKEN"] = spec.token
        env["PYTHONUNBUFFERED"] = "1"
        for (key, value) in spec.extraEnvironment { env[key] = value }
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        state = .starting
        do {
            try proc.run()
        } catch {
            state = .failed(String(describing: error))
            throw error
        }
        process = proc
        pidHolder.pid = proc.processIdentifier

        // Continuously drain stderr so the backend never blocks on a full pipe.
        let errCollector = StderrCollector()
        let errHandle = UncheckedBox(errPipe.fileHandleForReading)
        stderrTask = Task.detached {
            do {
                for try await line in errHandle.value.bytes.lines {
                    await errCollector.append(line)
                }
            } catch {
                // stderr drained or cancelled — nothing to do.
            }
        }

        let outHandle = UncheckedBox(outPipe.fileHandleForReading)
        do {
            let port = try await readReadyPort(from: outHandle, timeout: readyTimeout)
            state = .running(port: port)
            return port
        } catch {
            let detail = await errCollector.text()
            stderrTask?.cancel()
            if proc.isRunning { proc.terminate() }
            process = nil
            pidHolder.pid = nil
            state = .failed(failureMessage(error, stderr: detail))
            throw error
        }
    }

    func stop() {
        stderrTask?.cancel()
        stderrTask = nil
        guard let proc = process else { return }
        process = nil
        pidHolder.pid = nil
        // `terminationStatus` raises an ObjC exception unless the process has
        // actually exited, so wait for it after signalling.
        if proc.isRunning {
            proc.terminate()
            proc.waitUntilExit()
        }
        state = .stopped(code: proc.terminationStatus)
    }

    /// Races the stdout line reader against a timeout.
    private func readReadyPort(
        from outHandle: UncheckedBox<FileHandle>,
        timeout: Duration
    ) async throws -> Int {
        try await withThrowingTaskGroup(of: Int?.self) { group in
            group.addTask {
                for try await line in outHandle.value.bytes.lines {
                    if let ready = ReadyLineParser.parse(line) {
                        return ready.port
                    }
                }
                return nil // EOF before a ready line
            }
            group.addTask {
                try await Task.sleep(for: timeout)
                throw BackendError.readyTimeout
            }

            let first = try await group.next()!
            group.cancelAll()
            guard let port = first else { throw BackendError.exitedBeforeReady }
            return port
        }
    }

    private func failureMessage(_ error: Error, stderr: String) -> String {
        let head = stderr.split(separator: "\n").suffix(8).joined(separator: "\n")
        return head.isEmpty ? String(describing: error) : "\(error)\n\(head)"
    }
}
