import Foundation
import Testing

@testable import TinyForge

/// Thread-safe capture of the last request the client sent.
private final class RequestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: URLRequest?
    var request: URLRequest? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

private struct FakeTransport: Transport {
    let recorder: RequestRecorder
    let status: Int
    let body: Data

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        recorder.request = request
        let response = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
        )!
        return (body, response)
    }
}

struct APIClientTests {
    private let base = URL(string: "http://127.0.0.1:9999")!

    @Test func healthHitsHealthPathWithoutAuth() async throws {
        let recorder = RequestRecorder()
        let body = Data(#"{"status":"ok","name":"tinyforge","version":"0.0.1"}"#.utf8)
        let client = APIClient(
            baseURL: base, token: "tok",
            transport: FakeTransport(recorder: recorder, status: 200, body: body)
        )

        let health = try await client.health()

        #expect(health.status == "ok")
        #expect(health.name == "tinyforge")
        #expect(recorder.request?.url?.path == "/health")
        #expect(recorder.request?.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test func runtimeSendsBearerTokenAndDecodes() async throws {
        let recorder = RequestRecorder()
        let body = Data(
            #"{"python_version":"3.13.12","platform":"darwin","machine":"arm64","engines":{"mlx":false,"torch":true}}"#
                .utf8)
        let client = APIClient(
            baseURL: base, token: "tok",
            transport: FakeTransport(recorder: recorder, status: 200, body: body)
        )

        let info = try await client.runtime()

        #expect(info.pythonVersion == "3.13.12")
        #expect(info.machine == "arm64")
        #expect(info.engines["torch"] == true)
        #expect(recorder.request?.url?.path == "/v1/runtime")
        #expect(recorder.request?.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
    }

    @Test func unauthorizedMapsToAPIError() async {
        let client = APIClient(
            baseURL: base, token: "tok",
            transport: FakeTransport(recorder: RequestRecorder(), status: 401, body: Data())
        )

        await #expect(throws: APIError.unauthorized) {
            _ = try await client.runtime()
        }
    }

    @Test func unexpectedStatusCarriesServerDetail() async {
        let body = Data(#"{"detail":"run abc not found"}"#.utf8)
        let client = APIClient(
            baseURL: base, token: "tok",
            transport: FakeTransport(recorder: RequestRecorder(), status: 404, body: body)
        )

        await #expect(throws: APIError.unexpectedStatus(404, detail: "run abc not found")) {
            _ = try await client.runStatus(id: "abc")
        }
    }

    @Test func unexpectedStatusErrorDescriptionPrefersDetail() {
        #expect(APIError.unexpectedStatus(500, detail: "boom detail").errorDescription == "boom detail")
        #expect(APIError.unexpectedStatus(503, detail: nil).errorDescription?.contains("503") == true)
    }

    @Test func defaultTransportHasBoundedTimeouts() {
        let config = URLSessionTransport().session.configuration
        // No 7-day resource default; a hung backend call cannot spin forever.
        #expect(config.timeoutIntervalForRequest <= 300)
        #expect(config.timeoutIntervalForResource <= 3600)
    }
}
