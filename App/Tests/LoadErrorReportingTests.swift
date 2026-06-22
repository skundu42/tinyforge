import Testing

@testable import TinyForge

@MainActor
struct LoadErrorReportingTests {
    private final class Probe: LoadErrorReporting {
        var loadError: String?
    }

    @Test func attemptReturnsValueAndLeavesErrorNilOnSuccess() async {
        let probe = Probe()
        let value = await probe.attempt("Load thing") { 42 }
        #expect(value == 42)
        #expect(probe.loadError == nil)
    }

    @Test func attemptRecordsMessageAndReturnsNilOnFailure() async {
        let probe = Probe()
        let value: Int? = await probe.attempt("Load thing") {
            throw APIError.notImplemented
        }
        #expect(value == nil)
        #expect(probe.loadError?.contains("Load thing failed") == true)
    }
}
