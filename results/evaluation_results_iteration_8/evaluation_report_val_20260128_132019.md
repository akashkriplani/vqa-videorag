# Evaluation Report: VAL Split

**Generated:** 2026-01-28 13:28:52
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics

| Metric      | Mean   | Std    | Median | Min    | Max    |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| Recall@5    | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10   | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.3000 | 0.2971 | 0.2000 | 0.0000 | 1.4000 |
| mAP         | 0.4333 | 0.3903 | 0.3056 | 0.0000 | 1.0000 |
| nDCG@10     | 0.5190 | 0.3912 | 0.4957 | 0.0000 | 1.0000 |

### Temporal Accuracy

| Metric             | Mean   | Std    | Median |
| ------------------ | ------ | ------ | ------ |
| IoU                | 0.0925 | 0.0756 | 0.0903 |
| Temporal Precision | 0.2307 | 0.2167 | 0.1775 |
| Temporal Recall    | 0.1601 | 0.1404 | 0.1319 |
| Temporal F1        | 0.1608 | 0.1226 | 0.1657 |

**Correct Video Retrieved Rate:** 91.49%

---

## 2. Context Curation Performance

| Metric                 | Mean  | Std    | Median |
| ---------------------- | ----- | ------ | ------ |
| Pass Rate              | 0.00% | 0.0000 | 0.00%  |
| Reduction Rate         | 0.00% | 0.0000 | 0.00%  |
| Avg Conflicts Detected | 0.00  | 0.00   | 0      |

---

## 3. Answer Generation Performance

| Metric                | Mean      | Std       | Median    |
| --------------------- | --------- | --------- | --------- |
| Confidence            | 82.01%    | 0.0566    | 84.50%    |
| Answer Length (words) | 147.5     | 38.5      | 160       |
| Tokens Used           | 775.1     | 64.6      | 796       |
| Cost per Query        | $0.000216 | $0.000036 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0203

---

## 4. Performance Timing

| Metric              | Mean | Std  | Median |
| ------------------- | ---- | ---- | ------ |
| Search Time (s)     | 0.25 | 0.11 | 0.20   |
| Generation Time (s) | 5.06 | 1.32 | 5.25   |
| Total Time (s)      | 5.31 | 1.33 | 5.52   |

**Total Evaluation Time:** 8.3 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 16.08% Temporal F1
- **Answer Quality:** 82.01% Confidence
- **Cost Efficiency:** $0.000216 per query
