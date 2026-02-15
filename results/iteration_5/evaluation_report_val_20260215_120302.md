# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:07:54
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 47.65% | 0.3067 | 61.56% |
| Answer Length (words) | 105.6 | 60.6 | 144 |
| Tokens Used | 539.5 | 342.6 | 765 |
| Cost per Query | $0.000143 | $0.000100 | $0.000206 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0134

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.42 | 0.34 | 0.39 |
| Generation Time (s) | 2.55 | 2.03 | 3.56 |
| Total Time (s) | 2.98 | 2.08 | 4.00 |

**Total Evaluation Time:** 4.7 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 13.48% Temporal F1
- **Answer Quality:** 47.65% Confidence
- **Cost Efficiency:** $0.000143 per query

