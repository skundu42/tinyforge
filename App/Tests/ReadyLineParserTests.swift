import Testing

@testable import TinyForge

struct ReadyLineParserTests {
    @Test func parsesValidReadyLine() {
        let ready = ReadyLineParser.parse(#"{"event": "ready", "port": 54956}"#)
        #expect(ready?.event == "ready")
        #expect(ready?.port == 54956)
    }

    @Test func capturesDevTokenWhenPresent() {
        let ready = ReadyLineParser.parse(#"{"event":"ready","port":5,"token":"abc"}"#)
        #expect(ready?.token == "abc")
    }

    @Test func rejectsNonReadyEvent() {
        #expect(ReadyLineParser.parse(#"{"event":"log","port":1}"#) == nil)
    }

    @Test func rejectsGarbageAndEmpty() {
        #expect(ReadyLineParser.parse("not json at all") == nil)
        #expect(ReadyLineParser.parse("") == nil)
    }
}
