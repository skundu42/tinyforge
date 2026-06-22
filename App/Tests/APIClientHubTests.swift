import Foundation
import Testing

@testable import TinyForge

private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: URLRequest?
    var request: URLRequest? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

private struct StubTransport: Transport {
    let recorder: Recorder
    let status: Int
    let body: Data

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        recorder.request = request
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
        return (body, response)
    }
}

private func client(_ recorder: Recorder, status: Int = 200, json: String) -> APIClient {
    APIClient(
        baseURL: URL(string: "http://127.0.0.1:9999")!,
        token: "tok",
        transport: StubTransport(recorder: recorder, status: status, body: Data(json.utf8))
    )
}

struct APIClientHubTests {
    @Test func searchModelsBuildsQueryAndDecodes() async throws {
        let recorder = Recorder()
        let json = """
        [{"id":"mlx-community/Llama-3.2-1B-4bit","gated":false,"private":false,"tags":["mlx"],"downloads":42}]
        """
        let api = client(recorder, json: json)

        let models = try await api.searchModels(query: "llama", sort: "downloads", limit: 5)

        #expect(models.count == 1)
        #expect(models[0].id == "mlx-community/Llama-3.2-1B-4bit")
        #expect(models[0].downloads == 42)
        let comps = URLComponents(url: recorder.request!.url!, resolvingAgainstBaseURL: false)!
        #expect(comps.path == "/v1/hub/models")
        let items = Dictionary(uniqueKeysWithValues: (comps.queryItems ?? []).map { ($0.name, $0.value) })
        #expect(items["query"] == "llama")
        #expect(items["sort"] == "downloads")
        #expect(items["limit"] == "5")
        #expect(recorder.request?.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
    }

    @Test func modelDetailKeepsSlashedRepoIdInPath() async throws {
        let recorder = Recorder()
        let api = client(recorder, json: #"{"id":"a/b","gated":false,"tags":[],"siblings":[],"readme":null}"#)
        _ = try await api.modelDetail(repoId: "a/b")
        #expect(recorder.request?.url?.path == "/v1/hub/models/a/b")
    }

    @Test func startDownloadPostsBodyAndDecodesProgress() async throws {
        let recorder = Recorder()
        let json = """
        {"id":"job1","repo_id":"a/b","repo_type":"model","total_bytes":100,"downloaded_bytes":0,"fraction":0.0,"state":"running","error":null,"local_path":null}
        """
        let api = client(recorder, json: json)

        let progress = try await api.startDownload(repoId: "a/b", repoType: "model")

        #expect(progress.id == "job1")
        #expect(progress.state == "running")
        #expect(recorder.request?.httpMethod == "POST")
        let sentBody = try JSONDecoder().decode([String: String].self, from: recorder.request!.httpBody!)
        #expect(sentBody["repo_id"] == "a/b")
        #expect(sentBody["repo_type"] == "model")
    }

    @Test func deleteCachedParsesFreedBytes() async throws {
        let recorder = Recorder()
        let api = client(recorder, json: #"{"freed_bytes":2048}"#)
        let freed = try await api.deleteCached(repoId: "a/b")
        #expect(freed == 2048)
        #expect(recorder.request?.httpMethod == "DELETE")
    }

    @Test func loginPostsTokenAndDecodesStatus() async throws {
        let recorder = Recorder()
        let api = client(recorder, json: #"{"logged_in":true,"name":"alice"}"#)
        let status = try await api.login(token: "hf_secret")
        #expect(status.loggedIn == true)
        #expect(status.name == "alice")
        let sentBody = try JSONDecoder().decode([String: String].self, from: recorder.request!.httpBody!)
        #expect(sentBody["token"] == "hf_secret")
    }
}
