# Evaluation Report: VAL Split

**Generated:** 2026-01-28 16:04:18
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7234 | 0.4473 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.3021 | 0.3014 | 0.2000 | 0.0000 | 1.4000 |
| mAP | 0.4329 | 0.3894 | 0.3056 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5190 | 0.3910 | 0.4957 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0916 | 0.0763 | 0.0888 |
| Temporal Precision | 0.2284 | 0.2174 | 0.1730 |
| Temporal Recall | 0.1593 | 0.1413 | 0.1282 |
| Temporal F1 | 0.1593 | 0.1233 | 0.1631 |

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
| Confidence | 81.99% | 0.0562 | 84.32% |
| Answer Length (words) | 148.6 | 36.4 | 160 |
| Tokens Used | 776.2 | 62.3 | 796 |
| Cost per Query | $0.000217 | $0.000035 | $0.000232 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0204

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.14 | 0.23 |
| Generation Time (s) | 5.33 | 1.43 | 5.53 |
| Total Time (s) | 5.62 | 1.45 | 5.78 |

**Total Evaluation Time:** 8.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 15.93% Temporal F1
- **Answer Quality:** 81.99% Confidence
- **Cost Efficiency:** $0.000217 per query

