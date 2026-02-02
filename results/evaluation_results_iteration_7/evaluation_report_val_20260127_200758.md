# Evaluation Report: VAL Split

**Generated:** 2026-01-27 20:17:01
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7021 | 0.4573 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7021 | 0.4573 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2128 | 0.2017 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.4510 | 0.4034 | 0.3333 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5225 | 0.4010 | 0.5125 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0811 | 0.0708 | 0.0679 |
| Temporal Precision | 0.2048 | 0.1978 | 0.1549 |
| Temporal Recall | 0.1401 | 0.1324 | 0.1186 |
| Temporal F1 | 0.1424 | 0.1158 | 0.1272 |

**Correct Video Retrieved Rate:** 95.74%

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
| Confidence | 82.47% | 0.0535 | 84.63% |
| Answer Length (words) | 144.6 | 46.7 | 162 |
| Tokens Used | 765.9 | 72.1 | 796 |
| Cost per Query | $0.000211 | $0.000043 | $0.000232 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0198

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.13 | 0.22 |
| Generation Time (s) | 5.35 | 1.75 | 5.75 |
| Total Time (s) | 5.63 | 1.77 | 6.00 |

**Total Evaluation Time:** 8.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 14.24% Temporal F1
- **Answer Quality:** 82.47% Confidence
- **Cost Efficiency:** $0.000211 per query

