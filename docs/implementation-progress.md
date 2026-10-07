# Execution ledger: 2026-10-07 cardiovascular ML plan

Plan: docs/superpowers/plans/2026-10-07-cardiovascular-ml.md

User authorized execution with "continue to develop the model".
Ruling: work in the requested folder on feat/cardiovascular-ml; the repository has no initial commit, so a linked worktree cannot yet reference a base commit. Cost: isolation is by branch rather than a separate checkout.
Ruling: use a durable tracked ledger and native PowerShell commands instead of POSIX skill helpers. Cost: task bookkeeping is maintained directly.
Pre-flight: data/schema -> preprocessing -> train/evaluate -> artifacts/explain -> API interfaces checked; all targets use the same feature boundary and saved schema. The plan's explanation path explicitly separates base score and calibrated probability.
Task 1: complete. Dataset/schema tests observed RED (missing modules), then GREEN: 14 passed. Audit confirms exact source hash, 303 records, 55 inputs and one preserved discrepancy.
Task 2: complete. Pipeline tests observed RED, then GREEN. Full suite: 20 passed. Saved partition: 242 development / 61 test. Constant filtering and imputation fitted within each training pipeline.
Ruling: numerical fields are median-imputed for both families, with categorical missing sentinel; this is permitted by the plan's explicit missing-value policy and simplifies predictable partial-input handling. Cost: CatBoost native missing-value splits are not explored in this baseline.
Tasks 3-4: code/tests complete; full 60-candidate development experiment running (4 targets x 15 settings x 5 folds).
Task 5: artifact contract RED -> GREEN, including round-trip and evaluation overwrite protection; final actual bundle and evaluation pending training.
Task 6: explanations RED -> GREEN for both model families with calibration, additive full vectors, original-field aggregation and missing-input flags. Actual model explanation verification pending training.
Task 7: API RED -> GREEN, startup fails on missing bundle, 55-field schema and 422 responses covered. Actual server smoke test pending training.
Final review: independent read-only reviewer found two Important issues, no current data leakage: huge integers caused 500; saved splits lacked integrity validation.
Final fix pass: regression tests observed 10 failures before fixes; finite numeric overflow now becomes validation error; loaded partitions validate size/type/uniqueness/coverage/seed/class support and, when labels are supplied, match the deterministic split exactly.
Additional verification: independent four-target fitting and nested subset membership test added; all fits remain development-only.
Final: minor (deferred): Starlette's TestClient emits a dependency deprecation warning for httpx; runtime inference is unaffected. The lockfile records the tested version.
Final: Ruling: synthetic metadata lives in README and filename, not extra request fields, because the request schema intentionally forbids unknown fields. Cost: fixture consumers must read its documentation for context.
Task 8: reproducibility docs and lockfile drafted; final results/latency pending training.
Final fixes verified: full suite 57 passed, with one third-party TestClient deprecation warning. No test failures.
Repository guide agent.md appeared during execution and was read/preserved; API/inference test inputs now use the documented synthetic example. Source workbook remains the training/audit input only.
Ruling: related tasks were committed as coherent milestones rather than one commit per checkbox; the root repository initially had no history. Cost: commit boundaries combine several dependent components; tests and ledger preserve their checks.
