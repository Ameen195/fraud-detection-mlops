# Monitoring replay (Phase 6)

Validation + test replayed as weekly production batches. Static model: fraudguard v2. Label delay 30 days. Drift: Evidently on the top 20 SHAP features vs 30000 random train rows; score PSI vs week 0. Alarm if drift share >= 0.3 or score PSI >= 0.1. PR-AUC uses the eventual true labels; 'labels at' is the week those labels became known.

| week | split | days | rows | fraud rate | drift share | score PSI | alarm | labels at | PR-AUC static | PR-AUC retrain | live model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | valid | 101-106 | 14,759 | 0.0353 | 0.15 | 0.000 | no | 5 | 0.6288 | 0.6288 | v1 |
| 1 | valid | 106-113 | 20,875 | 0.0412 | 0.15 | 0.002 | no | 6 | 0.5481 | 0.5481 | v1 |
| 2 | valid | 113-120 | 21,101 | 0.0515 | 0.15 | 0.012 | no | 7 | 0.5755 | 0.5755 | v1 |
| 3 | valid | 120-127 | 23,618 | 0.0343 | 0.25 | 0.004 | no | 8 | 0.5889 | 0.5889 | v1 |
| 4 | valid | 127-134 | 19,102 | 0.0382 | 0.20 | 0.003 | no | 9 | 0.4937 | 0.4937 | v1 |
| 5 | valid | 134-141 | 18,653 | 0.0324 | 0.20 | 0.005 | no | 10 | 0.4356 | 0.4356 | v1 |
| 6 | test | 141-148 | 18,562 | 0.0345 | 0.25 | 0.003 | no | 11 | 0.4024 | 0.4024 | v1 |
| 7 | test | 148-155 | 21,476 | 0.0309 | 0.20 | 0.003 | no | after replay | 0.4900 | 0.4955 | v2 |
| 8 | test | 155-162 | 19,721 | 0.0284 | 0.15 | 0.002 | no | after replay | 0.4155 | 0.4213 | v2 |
| 9 | test | 162-169 | 21,096 | 0.0351 | 0.20 | 0.007 | no | after replay | 0.4456 | 0.4459 | v2 |
| 10 | test | 169-176 | 19,928 | 0.0373 | 0.25 | 0.006 | no | after replay | 0.5361 | 0.5260 | v2 |
| 11 | test | 176-183 | 17,325 | 0.0414 | 0.25 | 0.005 | no | after replay | 0.4661 | 0.4782 | v3 |

| pooled | weeks | alarm weeks | mean drift share | model | PR-AUC | recall@1%FPR | recall@5%FPR |
|---|---|---|---|---|---|---|---|
| valid | 6 | 0 | 0.18 | static | 0.5477 | 0.4487 | 0.6484 |
| valid | 6 | 0 | 0.18 | retrain | 0.5477 | 0.4487 | 0.6484 |
| test | 6 | 0 | 0.22 | static | 0.4593 | 0.3824 | 0.5854 |
| test | 6 | 0 | 0.22 | retrain | 0.4620 | 0.3797 | 0.5795 |

## Retrain decisions

- end of week 3 (schedule): skipped
- end of week 4 (schedule): skipped
- end of week 5 (schedule): skipped
- end of week 6 (schedule): promoted; challenger v2 PR-AUC 0.6009 vs champion v1 0.5940 on days 104-118 (trained up to day 104)
- end of week 10 (schedule): promoted; challenger v3 PR-AUC 0.5167 vs champion v2 0.4427 on days 132-146 (trained up to day 132)
