# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:16:01
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.5106 | 0.4999 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.5106 | 0.4999 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1787 | 0.2342 | 0.2000 | 0.0000 | 1.2000 |
| mAP | 0.2770 | 0.3652 | 0.1000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3400 | 0.3839 | 0.2891 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0768 | 0.0715 | 0.0622 |
| Temporal Precision | 0.1876 | 0.2088 | 0.1149 |
| Temporal Recall | 0.1414 | 0.1455 | 0.1086 |
| Temporal F1 | 0.1348 | 0.1178 | 0.1172 |

**Correct Video Retrieved Rate:** 85.11%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 87.45% | 0.0436 | 90.00% |
| Avg Conflicts Detected | 6.48 | 5.01 | 6 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 30.43% | 0.1554 | 30.00% |
| Answer Length (words) | 107.2 | 70.3 | 147 |
| Tokens Used | 483.1 | 186.8 | 570 |
| Cost per Query | $0.000136 | $0.000074 | $0.000184 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0128

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.22 | 0.09 | 0.19 |
| Generation Time (s) | 7.24 | 3.24 | 8.59 |
| Total Time (s) | 7.46 | 3.25 | 8.77 |

**Total Evaluation Time:** 11.7 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 13.48% Temporal F1
- **Answer Quality:** 30.43% Confidence
- **Cost Efficiency:** $0.000136 per query

