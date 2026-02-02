# Evaluation Report: TEST Split

**Generated:** 2026-01-27 20:26:53
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6765 | 0.4678 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6765 | 0.4678 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1961 | 0.1726 | 0.2000 | 0.0000 | 0.6000 |
| mAP | 0.4023 | 0.3870 | 0.2917 | 0.0000 | 1.0000 |
| nDCG@10 | 0.4758 | 0.3894 | 0.4826 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0815 | 0.0722 | 0.0667 |
| Temporal Precision | 0.1921 | 0.1802 | 0.1573 |
| Temporal Recall | 0.1358 | 0.1250 | 0.1184 |
| Temporal F1 | 0.1428 | 0.1189 | 0.1250 |

**Correct Video Retrieved Rate:** 97.06%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 82.52% | 0.0598 | 84.85% |
| Answer Length (words) | 137.1 | 55.3 | 161 |
| Tokens Used | 756.1 | 83.5 | 794 |
| Cost per Query | $0.000203 | $0.000050 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0207

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.27 | 0.13 | 0.21 |
| Generation Time (s) | 5.04 | 1.90 | 5.70 |
| Total Time (s) | 5.31 | 1.92 | 5.95 |

**Total Evaluation Time:** 9.0 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 14.28% Temporal F1
- **Answer Quality:** 82.52% Confidence
- **Cost Efficiency:** $0.000203 per query

