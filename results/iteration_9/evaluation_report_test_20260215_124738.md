# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:52:48
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics

| Metric      | Mean   | Std    | Median | Min    | Max    |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| Recall@5    | 0.6569 | 0.4748 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10   | 0.6569 | 0.4748 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2353 | 0.2350 | 0.2000 | 0.0000 | 1.0000 |
| mAP         | 0.2828 | 0.3459 | 0.1429 | 0.0000 | 1.0000 |
| nDCG@10     | 0.3783 | 0.3512 | 0.3333 | 0.0000 | 1.0000 |

### Temporal Accuracy

| Metric             | Mean   | Std    | Median |
| ------------------ | ------ | ------ | ------ |
| IoU                | 0.0921 | 0.0832 | 0.0820 |
| Temporal Precision | 0.2077 | 0.1965 | 0.1572 |
| Temporal Recall    | 0.1528 | 0.1328 | 0.1403 |
| Temporal F1        | 0.1586 | 0.1315 | 0.1516 |

**Correct Video Retrieved Rate:** 94.12%

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
| Confidence            | 48.78%    | 0.2922    | 55.48%    |
| Answer Length (words) | 104.8     | 62.9      | 145       |
| Tokens Used           | 561.5     | 322.3     | 766       |
| Cost per Query        | $0.000145 | $0.000095 | $0.000205 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0148

---

## 4. Performance Timing

| Metric              | Mean | Std  | Median |
| ------------------- | ---- | ---- | ------ |
| Search Time (s)     | 0.27 | 0.12 | 0.22   |
| Generation Time (s) | 2.65 | 2.07 | 3.70   |
| Total Time (s)      | 2.92 | 2.08 | 3.91   |

**Total Evaluation Time:** 5.0 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.86% Temporal F1
- **Answer Quality:** 48.78% Confidence
- **Cost Efficiency:** $0.000145 per query
