# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:59:59
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6667 | 0.4714 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6667 | 0.4714 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2471 | 0.2569 | 0.2000 | 0.0000 | 1.4000 |
| mAP | 0.2895 | 0.3364 | 0.1667 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3860 | 0.3453 | 0.3634 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0930 | 0.0764 | 0.0805 |
| Temporal Precision | 0.2082 | 0.1955 | 0.1678 |
| Temporal Recall | 0.1613 | 0.1385 | 0.1437 |
| Temporal F1 | 0.1615 | 0.1245 | 0.1490 |

**Correct Video Retrieved Rate:** 93.14%

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
| Confidence | 45.47% | 0.2957 | 53.32% |
| Answer Length (words) | 106.6 | 61.5 | 144 |
| Tokens Used | 546.8 | 336.9 | 768 |
| Cost per Query | $0.000144 | $0.000098 | $0.000206 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0147

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.18 | 0.22 |
| Generation Time (s) | 2.68 | 2.15 | 3.66 |
| Total Time (s) | 2.96 | 2.16 | 4.04 |

**Total Evaluation Time:** 5.0 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 16.15% Temporal F1
- **Answer Quality:** 45.47% Confidence
- **Cost Efficiency:** $0.000144 per query

