# Evaluation Report: VAL Split

**Generated:** 2026-01-20 01:59:18
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6702 | 0.4701 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6702 | 0.4701 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1638 | 0.1515 | 0.2000 | 0.0000 | 0.8000 |
| mAP | 0.4588 | 0.4306 | 0.3333 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5131 | 0.4206 | 0.5000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0785 | 0.0716 | 0.0584 |
| Temporal Precision | 0.1961 | 0.1862 | 0.1359 |
| Temporal Recall | 0.1362 | 0.1412 | 0.1037 |
| Temporal F1 | 0.1378 | 0.1170 | 0.1104 |

**Correct Video Retrieved Rate:** 91.49%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 86.70% | 0.0470 | 90.00% |
| Avg Conflicts Detected | 3.78 | 2.84 | 3 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 35.93% | 0.1473 | 30.00% |
| Answer Length (words) | 125.9 | 66.4 | 161 |
| Tokens Used | 555.0 | 118.4 | 588 |
| Cost per Query | $0.000162 | $0.000060 | $0.000197 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0153

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.15 | 0.04 | 0.14 |
| Generation Time (s) | 8.67 | 3.31 | 9.74 |
| Total Time (s) | 8.82 | 3.32 | 9.89 |

**Total Evaluation Time:** 13.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 13.78% Temporal F1
- **Answer Quality:** 35.93% Confidence
- **Cost Efficiency:** $0.000162 per query

