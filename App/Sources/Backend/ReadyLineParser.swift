import Foundation

/// Parses the backend's stdout readiness line.
enum ReadyLineParser {
    /// Returns the parsed ReadyLine, or nil if the line isn't a valid ready announcement.
    static func parse(_ line: String) -> ReadyLine? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let data = trimmed.data(using: .utf8),
            let ready = try? JSONDecoder().decode(ReadyLine.self, from: data),
            ready.event == "ready"
        else {
            return nil
        }
        return ready
    }
}
