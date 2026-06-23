import Testing

@testable import TinyForge

struct SystemMemoryTests {
    private let ram: UInt64 = 16_000_000_000  // 16 GB; 80% = 12.8 GB

    @Test func sizeOver80PercentIsTooBig() {
        #expect(exceedsMemoryBudget(sizeBytes: 13_000_000_000, physicalMemory: ram))
    }

    @Test func sizeUnder80PercentFits() {
        #expect(!exceedsMemoryBudget(sizeBytes: 8_000_000_000, physicalMemory: ram))
    }

    @Test func sizeExactlyAtThresholdFits() {
        // Strictly greater than the budget is "too big"; equal is fine.
        #expect(!exceedsMemoryBudget(sizeBytes: 12_800_000_000, physicalMemory: ram))
    }

    @Test func zeroOrUnknownSizeIsNeverTooBig() {
        #expect(!exceedsMemoryBudget(sizeBytes: 0, physicalMemory: ram))
    }

    @Test func customFractionIsRespected() {
        // At a 50% budget, 9 GB on a 16 GB machine is too big.
        #expect(exceedsMemoryBudget(sizeBytes: 9_000_000_000, physicalMemory: ram, fraction: 0.5))
    }
}
