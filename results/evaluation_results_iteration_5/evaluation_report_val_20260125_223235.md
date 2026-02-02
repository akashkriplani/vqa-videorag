# Evaluation Report: VAL Split

**Generated:** 2026-01-25 22:42:08
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 81.97% | 0.0564 | 84.32% |
| Answer Length (words) | 149.1 | 38.2 | 160 |
| Tokens Used | 776.5 | 63.0 | 794 |
| Cost per Query | $0.000217 | $0.000036 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0204

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.30 | 0.28 | 0.20 |
| Generation Time (s) | 5.65 | 1.55 | 5.79 |
| Total Time (s) | 5.95 | 1.57 | 6.11 |

**Total Evaluation Time:** 9.3 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 15.82% Temporal F1
- **Answer Quality:** 81.97% Confidence
- **Cost Efficiency:** $0.000217 per query

