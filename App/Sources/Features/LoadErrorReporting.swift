import Foundation

/// Adopted by view models that run background "load" calls — lists, caches,
/// status refreshes — whose failures must not vanish silently.
///
/// It replaces `(try? await api.foo()) ?? fallback`, which hides backend or
/// network failures behind empty UI, with
/// `await attempt("Load X") { try await api.foo() } ?? fallback`, which keeps the
/// graceful fallback *and* records a user-facing message in `loadError`.
///
/// `attempt` only ever *sets* `loadError` (on failure); it never clears it, so a
/// failure early in a multi-step load isn't wiped by a later success in the same
/// sequence. Clearing is an explicit UI action — the dismiss control on
/// `ErrorBanner` sets `loadError` back to `nil`.
@MainActor
protocol LoadErrorReporting: AnyObject {
    /// The most recent load failure, or `nil`. Surfaced by the UI via `ErrorBanner`.
    var loadError: String? { get set }
}

extension LoadErrorReporting {
    /// Runs a throwing load. Returns its value on success; on failure records a
    /// `"<what> failed: …"` message in `loadError` and returns `nil` so callers
    /// can fall back to a sensible default.
    @discardableResult
    func attempt<T>(_ what: String, _ operation: () async throws -> T) async -> T? {
        do {
            return try await operation()
        } catch {
            loadError = "\(what) failed: \(error)"
            return nil
        }
    }
}
