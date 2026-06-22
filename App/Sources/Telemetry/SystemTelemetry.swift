import Foundation
import Metal

/// A point-in-time snapshot of system signals relevant during training.
///
/// Note: GPU memory used by training lives in the *backend* process, so the
/// app's own `currentAllocatedSize` isn't the training figure (the worker
/// reports its peak memory via metrics). What the app can read system-wide is
/// the GPU memory budget and the thermal state.
struct SystemSnapshot: Sendable, Equatable {
    let gpuBudgetGB: Double
    let thermal: String
}

enum SystemTelemetry {
    static func sample() -> SystemSnapshot {
        let budgetBytes = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0
        return SystemSnapshot(
            gpuBudgetGB: Double(budgetBytes) / 1_073_741_824.0,
            thermal: thermalDescription(ProcessInfo.processInfo.thermalState)
        )
    }

    private static func thermalDescription(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }
}
