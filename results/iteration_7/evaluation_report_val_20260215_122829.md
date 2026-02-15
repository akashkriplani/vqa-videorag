# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:34:01
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.4574 | 0.4982 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.4574 | 0.4982 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1234 | 0.1679 | 0.0000 | 0.0000 | 1.0000 |
| mAP | 0.2993 | 0.3921 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3407 | 0.4058 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0716 | 0.0669 | 0.0568 |
| Temporal Precision | 0.1740 | 0.1765 | 0.1120 |
| Temporal Recall | 0.1232 | 0.1172 | 0.1071 |
| Temporal F1 | 0.1268 | 0.1106 | 0.1075 |

**Correct Video Retrieved Rate:** 75.53%

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
| Confidence | 57.96% | 0.2402 | 64.46% |
| Answer Length (words) | 118.6 | 61.9 | 154 |
| Tokens Used | 651.4 | 263.4 | 790 |
| Cost per Query | $0.000172 | $0.000083 | $0.000229 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0161

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.15 | 0.22 |
| Generation Time (s) | 3.11 | 1.93 | 3.88 |
| Total Time (s) | 3.39 | 1.95 | 4.06 |

**Total Evaluation Time:** 5.3 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 12.68% Temporal F1
- **Answer Quality:** 57.96% Confidence
- **Cost Efficiency:** $0.000172 per query

