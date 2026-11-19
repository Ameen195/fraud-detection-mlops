# FraudGuard

Real-time fraud detection for digital payment transactions: train, track, serve, monitor, retrain.

Status: Phase 1 (setup). Full README comes in Phase 9.

Data note: IEEE-CIS is card/e-commerce payment data, not UPI.

## Serving (Phase 5)

`docker compose up` starts the API on :8000 (`/score`, `/score/batch`, `/health`, `/model`) and the
MLflow UI on :5000. Inputs (registry, `data/processed`, `eval/results`) are mounted read-only.
Latency: `eval/results/latency.md`.

Scoring needs each card's past transactions, so the API keeps the processed history in memory
(never updated by scoring). Production would use a feature store (online per-card state) instead.

Known limit: `/score` latency is p50 281 ms, p95 531 ms (300 validation rows, Docker, one request
at a time). Cause: common emails and devices match a large share of the history, so
`build_features` runs on a big row subset per request. Fix: precompute the per-card, per-value
counts and window sums in a feature store and read them at request time. This is not done here
because it would duplicate the feature logic that training and serving share.
