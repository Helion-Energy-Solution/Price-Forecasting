Add a new experiment entry to RESEARCH_LOG.md for the Price Forecasting project.

## Steps

1. Read `RESEARCH_LOG.md` to find the last EXP number and understand the current log structure.

2. Read `data/processed/backtest_latest.json` to get the latest backtest results (written automatically by the revenue backtest cell in the notebook). This file contains the date the backtest was run and the full KPI table — no copy-paste needed.

3. Read the pinball metric files written by each model script after training:
   - `models/trl_weekly/pinball_latest.json` — per (direction, quantile): pinball_es (fixed es_val window), pinball_kpi (growing kpi_val window), n_estimators
   - `models/trl_daily/pinball_latest.json` — per (direction, quantile): pinball_kpi
   - `models/tre/pinball_latest.json` — per (direction, quantile): pinball_kpi, clf_auc
   If a file does not exist (model not retrained this run), note that it was unchanged.

4. Read `config/config.yaml`, `src/models/trl_weekly_model.py`, `src/models/trl_daily_model.py`, `src/models/tre_model.py`, and `src/data/feature_store.py` to understand what changed since the last experiment entry — focus on FEATURE_COLS, model parameters, val splits, and feature engineering.

5. Determine the next EXP number (last EXP + 1).

6. Append a new experiment section to RESEARCH_LOG.md following the exact format of existing entries:
   - `### EXP-NNN — <short title describing what changed>`
   - `**Date:**` from backtest_latest.json
   - `**Changes from EXP-NNN-1:**` bullet list of what changed (inferred from reading the code)
   - `**Motivation:**` why the change was made (infer from the nature of the change, or ask the user if unclear)
   - Revenue KPI table with columns: Market, capture_%, opt_select_%, opt_pnl/slot, oracle_pnl/slot, Δ capture vs previous EXP
   - Pinball table with columns: Model, direction, q10, q25, q50, q75, q90 (pinball_es for TRL Weekly, pinball_kpi for all). Only include rows for models that were retrained this run. For TRL Weekly include both es and kpi rows. For TRE also include clf_auc per direction.
   - `**Notes:**` interpretation — flag small val sets (TRL Weekly ≈ 8 slots, TRE neg extreme ≈ few slots), note which improvements are reliable vs potentially noisy. Note that TRL Weekly pinball_es is the most stable comparison (fixed window); pinball_kpi grows with the val set.

7. Check if any new design decisions or assumptions should be added to **Key Assumptions** or **Design Decisions**, and update if needed.

8. If any open questions were resolved, update their status in the **Open Questions** table.

9. If the user passed additional context or notes via `$ARGUMENTS`, incorporate them into the entry.

Additional context from user (if any):
$ARGUMENTS
