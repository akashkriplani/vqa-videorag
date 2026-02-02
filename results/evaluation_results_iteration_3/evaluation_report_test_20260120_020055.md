# Evaluation Report: TEST Split

**Generated:** 2026-01-20 02:15:28
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics

| Metric      | Mean   | Std    | Median | Min    | Max    |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| Recall@5    | 0.5588 | 0.4965 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10   | 0.5588 | 0.4965 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1373 | 0.1400 | 0.2000 | 0.0000 | 0.6000 |
| mAP         | 0.3564 | 0.4069 | 0.1548 | 0.0000 | 1.0000 |
| nDCG@10     | 0.4099 | 0.4142 | 0.3448 | 0.0000 | 1.0000 |

### Temporal Accuracy

| Metric             | Mean   | Std    | Median |
| ------------------ | ------ | ------ | ------ |
| IoU                | 0.0671 | 0.0650 | 0.0564 |
| Temporal Precision | 0.1706 | 0.1835 | 0.1119 |
| Temporal Recall    | 0.1086 | 0.1009 | 0.0879 |
| Temporal F1        | 0.1192 | 0.1084 | 0.1068 |

**Correct Video Retrieved Rate:** 89.22%

---

## 2. Context Curation Performance

| Metric                 | Mean    | Std    | Median  |
| ---------------------- | ------- | ------ | ------- |
| Pass Rate              | 100.00% | 0.0000 | 100.00% |
| Reduction Rate         | 85.20%  | 0.0519 | 90.00%  |
| Avg Conflicts Detected | 3.87    | 4.32   | 2       |

---

## 3. Answer Generation Performance

| Metric                | Mean      | Std       | Median    |
| --------------------- | --------- | --------- | --------- |
| Confidence            | 32.43%    | 0.1594    | 30.00%    |
| Answer Length (words) | 121.4     | 68.2      | 164       |
| Tokens Used           | 562.4     | 118.2     | 592       |
| Cost per Query        | $0.000159 | $0.000061 | $0.000195 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0162

---

## 4. Performance Timing

| Metric              | Mean | Std  | Median |
| ------------------- | ---- | ---- | ------ |
| Search Time (s)     | 0.15 | 0.04 | 0.14   |
| Generation Time (s) | 8.21 | 3.18 | 9.12   |
| Total Time (s)      | 8.36 | 3.18 | 9.25   |

**Total Evaluation Time:** 14.2 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 11.92% Temporal F1
- **Answer Quality:** 32.43% Confidence
- **Cost Efficiency:** $0.000159 per query
