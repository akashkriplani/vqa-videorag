# Evaluation Report: TEST Split

**Generated:** 2026-02-15 11:52:17
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.3529 | 0.4779 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.3529 | 0.4779 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0765 | 0.1086 | 0.0000 | 0.0000 | 0.4000 |
| mAP | 0.2706 | 0.4153 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.2910 | 0.4216 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0704 | 0.0748 | 0.0560 |
| Temporal Precision | 0.1770 | 0.1991 | 0.1276 |
| Temporal Recall | 0.1094 | 0.1097 | 0.0922 |
| Temporal F1 | 0.1231 | 0.1218 | 0.1061 |

**Correct Video Retrieved Rate:** 63.73%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 85.20% | 0.0537 | 90.00% |
| Avg Conflicts Detected | 5.84 | 4.44 | 5 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 25.59% | 0.1772 | 30.00% |
| Answer Length (words) | 98.8 | 68.5 | 48 |
| Tokens Used | 455.9 | 232.4 | 502 |
| Cost per Query | $0.000124 | $0.000080 | $0.000095 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0127

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.69 | 1.59 | 0.19 |
| Generation Time (s) | 10.06 | 8.89 | 7.98 |
| Total Time (s) | 10.75 | 9.69 | 8.20 |

**Total Evaluation Time:** 18.3 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 12.31% Temporal F1
- **Answer Quality:** 25.59% Confidence
- **Cost Efficiency:** $0.000124 per query

