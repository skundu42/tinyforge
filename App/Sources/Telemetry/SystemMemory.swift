import Foundation

/// True when a model of `sizeBytes` would exceed `fraction` of the machine's RAM
/// — a heuristic for "won't comfortably fit in memory". File size is the only
/// pre-download signal and is a reasonable lower bound for the load footprint.
func exceedsMemoryBudget(sizeBytes: Int, physicalMemory: UInt64, fraction: Double = 0.8) -> Bool {
    guard sizeBytes > 0, physicalMemory > 0 else { return false }
    return Double(sizeBytes) > fraction * Double(physicalMemory)
}

/// The host machine's physical memory and memory-budget checks built on it.
enum SystemMemory {
    static var physicalMemory: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// Human-readable total RAM, e.g. "16 GB" (for warning tooltips).
    static var physicalMemoryString: String { ByteFormat.string(Int(physicalMemory)) }

    /// Whether a model of `sizeBytes` is too big for this machine (> 80% of RAM).
    /// `sizeBytes <= 0` (unknown size) is treated as "not too big".
    static func isTooBig(sizeBytes: Int?) -> Bool {
        guard let sizeBytes else { return false }
        return exceedsMemoryBudget(sizeBytes: sizeBytes, physicalMemory: physicalMemory)
    }
}

extension CachedRepo {
    /// Whether this cached model exceeds the system memory budget (> 80% of RAM).
    var isTooBigForSystem: Bool { SystemMemory.isTooBig(sizeBytes: sizeOnDisk) }

    /// Picker label that flags oversized models inline (menus can't host a chip).
    var pickerLabel: String { isTooBigForSystem ? "\(repoId) · too big" : repoId }
}
