# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:40:10
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.5106 | 0.4999 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.5106 | 0.4999 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1809 | 0.2353 | 0.2000 | 0.0000 | 1.2000 |
| mAP | 0.2734 | 0.3626 | 0.1000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3375 | 0.3814 | 0.2891 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0759 | 0.0716 | 0.0587 |
| Temporal Precision | 0.1858 | 0.2057 | 0.1243 |
| Temporal Recall | 0.1380 | 0.1360 | 0.1018 |
| Temporal F1 | 0.1333 | 0.1175 | 0.1109 |

**Correct Video Retrieved Rate:** 88.30%

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
| Confidence | 48.48% | 0.3047 | 62.25% |
| Answer Length (words) | 113.9 | 59.0 | 152 |
| Tokens Used | 558.8 | 343.3 | 786 |
| Cost per Query | $0.000152 | $0.000100 | $0.000223 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0143

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.31 | 0.13 | 0.29 |
| Generation Time (s) | 2.78 | 2.11 | 3.80 |
| Total Time (s) | 3.09 | 2.14 | 4.08 |

**Total Evaluation Time:** 4.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 13.33% Temporal F1
- **Answer Quality:** 48.48% Confidence
- **Cost Efficiency:** $0.000152 per query

