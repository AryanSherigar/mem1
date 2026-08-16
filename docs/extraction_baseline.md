# Extraction Baseline v1

## Status

Implemented deterministic fixture path only. This is an integration and data-contract baseline, not evidence that an extraction model is accurate, safe, calibrated, or production-ready.

## Guardrails

- Only extractor identity `deterministic-fixture` version `v1` is accepted.
- Every SQL attempt is stamped `extractor_kind=deterministic_fixture` and `quality_status=baseline_only`; the migration rejects other values. Its deterministic identity includes the source hash and fixture output, so changed fixture output makes a separate audit attempt rather than overwriting history.
- Immutable source chunks remain episodic evidence. Derived candidates are semantic by default; procedural needs an explicit fixture type.
- No automatic user or organization promotion. Default scope is session, otherwise chat.
- Invalid drafts are retained with their rejection reason. They are not silently dropped or converted into facts.

## Acceptance matrix

The source-controlled cases live in [fixtures/extraction_baseline_v1.json](fixtures/extraction_baseline_v1.json). Unit tests execute the corresponding critical conditions: Unicode offsets, scope defaults, explicit procedural classification, rejection retention, deterministic replay, changed-output audit separation, and a block on any non-fixture extractor.

## Production gate

A model-backed extractor requires a new approved ADR and milestone packet covering provider/privacy, structured-output contract, quality dataset and metrics, calibration, error analysis, cost/latency, versioning, rollback, and a migration that deliberately expands the baseline-only SQL constraint. It must not reuse this baseline status as a quality claim.
