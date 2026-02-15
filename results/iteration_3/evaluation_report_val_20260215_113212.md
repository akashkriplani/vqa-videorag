# Evaluation Report: VAL Split

**Generated:** 2026-02-15 11:50:32
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.3511 | 0.4773 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.3511 | 0.4773 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0809 | 0.1214 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.2658 | 0.4073 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.2880 | 0.4164 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0638 | 0.0699 | 0.0462 |
| Temporal Precision | 0.1651 | 0.1839 | 0.1099 |
| Temporal Recall | 0.1082 | 0.1393 | 0.0666 |
| Temporal F1 | 0.1123 | 0.1163 | 0.0882 |

**Correct Video Retrieved Rate:** 64.89%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 87.55% | 0.0430 | 90.00% |
| Avg Conflicts Detected | 4.03 | 2.85 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 29.42% | 0.2067 | 30.00% |
| Answer Length (words) | 100.6 | 69.9 | 87 |
| Tokens Used | 447.1 | 208.3 | 479 |
| Cost per Query | $0.000124 | $0.000076 | $0.000123 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0117

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.64 | 0.68 | 0.28 |
| Generation Time (s) | 10.77 | 8.14 | 9.29 |
| Total Time (s) | 11.41 | 8.52 | 9.58 |

**Total Evaluation Time:** 17.9 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 11.23% Temporal F1
- **Answer Quality:** 29.42% Confidence
- **Cost Efficiency:** $0.000124 per query

