# Test results (Phase 4): fraudguard@champion v2

Threshold 0.0198, chosen on validation (cost per false alarm = 5.0). Test set scored once.

| metric | value |
|---|---|
| pr_auc | 0.4593 |
| roc_auc | 0.8874 |
| recall_at_1pct_fpr | 0.3824 |
| recall_at_5pct_fpr | 0.5854 |
| precision @ threshold | 0.1182 |
| recall @ threshold | 0.8278 |

| business (test) | value |
|---|---|
| fraud amount caught | 504,553.85 |
| fraud amount missed | 105,380.48 |
| false alarms | 25,091 |
| false alarm rate (of 114,044 legitimate test transactions) | 22.00% |
| false alarm cost | 125,455.00 |
| net saving (caught - false alarm cost) | 379,098.85 |

Assumptions: the cost of 5 per false alarm assumes a flag triggers a cheap step-up check (like an OTP), not a blocked payment. If a flag blocks the payment, the real cost is higher; see the cost sensitivity table. Recall @5% FPR (0.5854) is the stricter operating point: it limits false alarms to 5% of legitimate transactions, versus the 22.00% at the chosen threshold.

## Cost sensitivity (validation only)

| cost per false alarm | threshold | recall | precision | false alarms |
|---|---|---|---|---|
| 1 | 0.0064 | 0.9751 | 0.0615 | 68,616 |
| 5 | 0.0198 | 0.8653 | 0.1412 | 24,272 |
| 20 | 0.0464 | 0.7358 | 0.2618 | 9,569 |
| 50 | 0.1296 | 0.5426 | 0.4774 | 2,739 |
