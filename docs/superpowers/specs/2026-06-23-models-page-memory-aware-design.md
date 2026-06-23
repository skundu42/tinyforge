# Models page: memory-aware reorder + "Too big for your system"

Date: 2026-06-23

## Goal

1. The Models page always shows **downloaded models first** (top), with search/browse
   below — instead of today's behaviour where searching replaces the downloaded list.
2. Warn when a model's size exceeds **80% of the system's physical RAM** with a
   tag reading **"Too big for your system"**.
3. Apply the warning consistently wherever a *sized* model is shown or selectable.

## Components

### 1. `SystemMemory` (App/Sources/Telemetry/SystemMemory.swift)
- Injectable physical-memory source (default `ProcessInfo.processInfo.physicalMemory`).
- Pure, unit-tested:
  `exceedsMemoryBudget(sizeBytes: Int, physicalMemory: UInt64, fraction: Double = 0.8) -> Bool`
  → `sizeBytes > 0 && Double(sizeBytes) > fraction * Double(physicalMemory)`.
- Convenience `isTooBig(sizeBytes: Int) -> Bool` using the real RAM value.

### 2. `TooBigTag` (DesignSystem)
- Warning-coloured capsule (`Theme.ember`) reading "Too big for your system".
- `.help(...)` tooltip explaining the 80%-of-RAM basis, so the warning isn't cryptic.

### 3. `HubBrowserView` — `VSplitView`
- **Top pane (always present):** downloaded-models panel; a slim "no downloads yet"
  hint when empty. `TooBigTag` on rows whose `CachedRepo.sizeOnDisk` exceeds budget.
- **Bottom pane:** the search bar + existing phase content (idle hint / spinner /
  error / results master-detail). `TooBigTag` in the detail view next to the total
  size and by the Download button (`HubModelDetail.totalSize`).
- Native draggable divider (resizable halves).

### 4. Finetune (`TrainingView`) & Playground (`PlaygroundView`) cached-model pickers
- A warning line beneath the picker when the *selected* cached model is too big.
- A "· too big" suffix on oversized items in the picker menu.
- Only the HF cached-model picker; from-scratch (`lm`) runs are tiny.

## Non-goals (YAGNI)
- No per-search-row size fetching (raw `HubModel` rows carry no size).
- No hard block on downloading/loading an oversized model — warn, don't prevent.
- No backend change; RAM is read natively in Swift.

## Testing
- Unit-test `exceedsMemoryBudget`: boundary at exactly 80%, zero size, over/under.
- Verify the UI via build + the existing Swift suite staying green (views aren't
  unit-tested in this codebase).
