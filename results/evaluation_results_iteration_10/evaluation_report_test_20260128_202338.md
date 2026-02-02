# Evaluation Report: TEST Split

**Generated:** 2026-01-28 20:31:36
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics

| Metric      | Mean   | Std    | Median | Min    | Max    |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| Recall@5    | 0.7157 | 0.4511 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10   | 0.7157 | 0.4511 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.3216 | 0.3083 | 0.2000 | 0.0000 | 1.4000 |
| mAP         | 0.3870 | 0.3629 | 0.2589 | 0.0000 | 1.0000 |
| nDCG@10     | 0.4817 | 0.3704 | 0.4585 | 0.0000 | 1.0000 |

### Temporal Accuracy

| Metric             | Mean   | Std    | Median |
| ------------------ | ------ | ------ | ------ |
| IoU                | 0.0980 | 0.0861 | 0.0845 |
| Temporal Precision | 0.2229 | 0.2213 | 0.1662 |
| Temporal Recall    | 0.1606 | 0.1321 | 0.1568 |
| Temporal F1        | 0.1679 | 0.1363 | 0.1558 |

**Correct Video Retrieved Rate:** 98.04%

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
| Confidence            | 81.48%    | 0.0688    | 84.27%    |
| Answer Length (words) | 139.9     | 51.4      | 160       |
| Tokens Used           | 764.3     | 80.8      | 798       |
| Cost per Query        | $0.000208 | $0.000048 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0212

---

## 4. Performance Timing

| Metric              | Mean | Std  | Median |
| ------------------- | ---- | ---- | ------ |
| Search Time (s)     | 0.26 | 0.11 | 0.21   |
| Generation Time (s) | 4.31 | 1.55 | 4.62   |
| Total Time (s)      | 4.57 | 1.55 | 4.93   |

**Total Evaluation Time:** 7.8 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 16.79% Temporal F1
- **Answer Quality:** 81.48% Confidence
- **Cost Efficiency:** $0.000208 per query
