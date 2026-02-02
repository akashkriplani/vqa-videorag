# Evaluation Report: VAL Split

**Generated:** 2026-01-28 20:21:50
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7340 | 0.4418 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7340 | 0.4418 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.3319 | 0.3011 | 0.2000 | 0.0000 | 1.4000 |
| mAP | 0.4230 | 0.3796 | 0.3268 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5153 | 0.3827 | 0.5252 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.1028 | 0.0866 | 0.0901 |
| Temporal Precision | 0.2439 | 0.2238 | 0.1830 |
| Temporal Recall | 0.1787 | 0.1630 | 0.1396 |
| Temporal F1 | 0.1759 | 0.1356 | 0.1653 |

**Correct Video Retrieved Rate:** 93.62%

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
| Confidence | 82.59% | 0.0513 | 84.32% |
| Answer Length (words) | 151.0 | 31.3 | 158 |
| Tokens Used | 784.6 | 53.9 | 798 |
| Cost per Query | $0.000222 | $0.000030 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0208

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.22 | 0.08 | 0.19 |
| Generation Time (s) | 5.10 | 1.17 | 5.27 |
| Total Time (s) | 5.32 | 1.18 | 5.51 |

**Total Evaluation Time:** 8.3 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 17.59% Temporal F1
- **Answer Quality:** 82.59% Confidence
- **Cost Efficiency:** $0.000222 per query

