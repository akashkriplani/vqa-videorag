# Evaluation Report: VAL Split

**Generated:** 2026-01-24 23:40:48
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.4255 | 0.4944 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.4255 | 0.4944 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1021 | 0.1422 | 0.0000 | 0.0000 | 0.8000 |
| mAP | 0.2314 | 0.3646 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.2783 | 0.3752 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0647 | 0.0657 | 0.0494 |
| Temporal Precision | 0.1751 | 0.1935 | 0.1326 |
| Temporal Recall | 0.1120 | 0.1366 | 0.0732 |
| Temporal F1 | 0.1147 | 0.1107 | 0.0941 |

**Correct Video Retrieved Rate:** 84.04%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 87.98% | 0.0402 | 90.00% |
| Avg Conflicts Detected | 4.05 | 2.97 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 27.35% | 0.1668 | 30.00% |
| Answer Length (words) | 98.3 | 73.1 | 66 |
| Tokens Used | 485.0 | 144.4 | 478 |
| Cost per Query | $0.000131 | $0.000067 | $0.000104 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0124

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.16 | 0.04 | 0.15 |
| Generation Time (s) | 7.45 | 3.40 | 6.34 |
| Total Time (s) | 7.61 | 3.40 | 6.55 |

**Total Evaluation Time:** 11.9 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 11.47% Temporal F1
- **Answer Quality:** 27.35% Confidence
- **Cost Efficiency:** $0.000131 per query

