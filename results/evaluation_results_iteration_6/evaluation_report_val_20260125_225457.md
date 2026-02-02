# Evaluation Report: VAL Split

**Generated:** 2026-01-25 23:07:34
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.3000 | 0.3014 | 0.2000 | 0.0000 | 1.4000 |
| mAP | 0.4346 | 0.3896 | 0.3333 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5199 | 0.3912 | 0.5000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0909 | 0.0752 | 0.0888 |
| Temporal Precision | 0.2275 | 0.2167 | 0.1730 |
| Temporal Recall | 0.1583 | 0.1402 | 0.1282 |
| Temporal F1 | 0.1582 | 0.1217 | 0.1631 |

**Correct Video Retrieved Rate:** 91.49%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 98.30% | 0.1107 | 100.00% |
| Reduction Rate | 87.98% | 0.0402 | 90.00% |
| Avg Conflicts Detected | 6.91 | 5.51 | 6 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 35.58% | 0.1376 | 30.00% |
| Answer Length (words) | 117.4 | 70.5 | 162 |
| Tokens Used | 525.2 | 116.6 | 580 |
| Cost per Query | $0.000151 | $0.000062 | $0.000191 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0142

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.16 | 0.06 | 0.15 |
| Generation Time (s) | 7.67 | 2.84 | 8.70 |
| Total Time (s) | 7.83 | 2.84 | 8.86 |

**Total Evaluation Time:** 12.3 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 15.82% Temporal F1
- **Answer Quality:** 35.58% Confidence
- **Cost Efficiency:** $0.000151 per query

