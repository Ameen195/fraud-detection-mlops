# Monitoring and retraining

Code: `src/fraudguard/pipelines/flows.py` (Prefect flows), `src/fraudguard/monitoring/`
(drift and label-timing rules). Settings: the `monitoring:` block in `config/settings.yaml`.
Results of the replay: `eval/results/monitoring.md`, `monitoring.json`, `replay_pr_auc.png`.

## The flows

| flow | does | inputs it may use |
|---|---|---|
| `monitor_batch` | scores a batch, Evidently drift on the top-20 SHAP features vs. a train sample, score PSI vs. a reference window, performance for batches whose labels have arrived | transactions up to now; labels at least `label_delay_days` old |
| `retrain` | trains a challenger with the champion's hyperparameters, compares both on the newest labeled `eval_window_days`, promotes only if PR-AUC beats the champion by `min_promotion_gain`; otherwise registers it with alias `candidate` | labels at least `label_delay_days` old |
| `replay` | runs the two flows above week by week over validation + test (the offline proof) | as above, enforced per week |

Rules that keep it honest:

- **Label delay.** A label for a transaction at time *t* is used only once `t + label_delay_days <= now`.
  With a 30-day delay and a 14-day eval window, the first retrain can only run ~44 days
  after the champion's training data ends. Before that, `retrain` is skipped and logged.
- **Clean comparison.** The eval window must lie entirely after the champion's training data
  (`train_end_dt` tag on each registry version), so the champion is never judged on data it saw.
- **Threshold-free promotion.** Promotion uses PR-AUC only; recall at 1%/5% FPR is logged next to it.
  No threshold is tuned on the eval window.
- Every decision is logged: an MLflow run per retrain (both models' metrics), registry aliases
  (`champion` / `candidate`), and the decision list in `monitoring.json`.

## How it would run on a schedule in production

Everything stays free and local: an open-source Prefect server, the MLflow SQLite registry, Docker.

1. Start the Prefect server (UI at http://127.0.0.1:4200), in the repo root:
   ```powershell
   uv run prefect server start
   ```
2. Point flows at it and serve them with schedules, in a second terminal:
   ```powershell
   $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
   ```
   A small script would call `prefect.serve(...)` with:
   - `monitor_batch.to_deployment(name="daily-monitor", cron="0 6 * * *")`: once a day, on the
     previous day's scored transactions (the replay uses weekly batches only to keep the table short).
   - `retrain.to_deployment(name="weekly-retrain", cron="0 7 * * 1")`: the scheduled retrain.
   - Drift-triggered retrain: `monitor_batch` calls `retrain` as a subflow when the alarm fires
     (what the replay does), or a Prefect Automation runs the `retrain` deployment on the
     alarm event.
3. Data in production: scored transactions are written by the API (or the Phase 8 stream consumer)
   to a table; chargeback labels are joined in as they arrive. `monitor_batch` and `retrain` read
   from that table instead of the replay parquet, and the label-delay rule becomes "only rows that
   have a label", plus the same minimum age so late chargebacks are not missed.
4. Rollout: the API loads `fraudguard@champion` at startup (`serving/app.py`). After a promotion,
   restart the API container (`docker compose restart api`) to pick up the new version. The
   previous champion stays in the registry, so rollback is moving the alias back.
5. Alerts: Prefect notifications (email/Slack blocks, free) on flow failure, on a drift alarm, and on
   every promotion.

Not built (out of scope for a local portfolio project): the scored-transaction table, the
chargeback feed, API hot-reload, and the `prefect.serve` script itself.

## Findings from the replay

All numbers below are from `eval/results/monitoring.json`. The alarm thresholds
(`drift_share_alarm`, `score_psi_alarm`) were set in config before the replay ran and were not
tuned afterwards.

- **Drift monitoring would not have detected the drop.** No week raised an alarm. Feature drift
  share stayed between 0.15 and 0.25 in every week (mean 0.18 in validation, 0.22 in test) and
  score PSI never exceeded 0.012. The inputs and score distribution barely moved while PR-AUC fell.
- **The drop started inside validation, not at the test boundary.** Weekly PR-AUC was 0.55-0.63
  in weeks 0-3, then 0.49 and 0.44 in weeks 4-5, and 0.40 in week 6 (first test week). Weekly
  fraud rate moved between 2.8% and 5.2%, so part of the swing is a changing base rate, which
  PR-AUC is sensitive to.
- **Labels are the only signal, and they are slow.** With a 30-day label delay, week 4's drop
  became measurable only at the end of week 9.
- **Retraining helped little.** The first retrain was possible only at the end of week 6
  (label delay + eval window); both challengers beat the champion on their eval windows and
  were promoted. Pooled test PR-AUC: 0.4593 static vs 0.4620 with retraining, while recall at
  1% and 5% FPR was slightly lower with retraining (0.3797 vs 0.3824, 0.5795 vs 0.5854).
- **Takeaway:** for this data, monitor labeled performance and the fraud rate as soon as labels
  arrive. Feature drift on the top features is not an early warning here.
