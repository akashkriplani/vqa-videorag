# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:34:15
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.4314 | 0.4953 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.4314 | 0.4953 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1137 | 0.1495 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.2578 | 0.3829 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3014 | 0.3945 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0879 | 0.0783 | 0.0728 |
| Temporal Precision | 0.1947 | 0.1892 | 0.1444 |
| Temporal Recall | 0.1542 | 0.1357 | 0.1235 |
| Temporal F1 | 0.1527 | 0.1249 | 0.1357 |

**Correct Video Retrieved Rate:** 80.39%

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
| Confidence | 56.37% | 0.2549 | 63.60% |
| Answer Length (words) | 105.4 | 63.6 | 148 |
| Tokens Used | 616.3 | 272.0 | 766 |
| Cost per Query | $0.000157 | $0.000085 | $0.000212 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0160

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.25 | 0.13 | 0.19 |
| Generation Time (s) | 2.84 | 1.98 | 3.71 |
| Total Time (s) | 3.09 | 2.01 | 3.99 |

**Total Evaluation Time:** 5.2 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.27% Temporal F1
- **Answer Quality:** 56.37% Confidence
- **Cost Efficiency:** $0.000157 per query

