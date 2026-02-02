# Evaluation Report: VAL Split

**Generated:** 2026-01-22 21:43:07
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7128 | 0.4525 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7128 | 0.4525 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1745 | 0.1494 | 0.2000 | 0.0000 | 0.8000 |
| mAP | 0.4633 | 0.4164 | 0.3583 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5266 | 0.4027 | 0.5254 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0810 | 0.0808 | 0.0570 |
| Temporal Precision | 0.2004 | 0.2028 | 0.1245 |
| Temporal Recall | 0.1354 | 0.1399 | 0.0941 |
| Temporal F1 | 0.1403 | 0.1285 | 0.1079 |

**Correct Video Retrieved Rate:** 94.68%

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
| Confidence | 83.98% | 0.0414 | 84.71% |
| Answer Length (words) | 138.2 | 54.2 | 158 |
| Tokens Used | 754.2 | 85.1 | 795 |
| Cost per Query | $0.000205 | $0.000050 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0192

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.22 | 0.08 | 0.18 |
| Generation Time (s) | 5.25 | 2.00 | 5.72 |
| Total Time (s) | 5.47 | 2.00 | 5.96 |

**Total Evaluation Time:** 8.6 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 14.03% Temporal F1
- **Answer Quality:** 83.98% Confidence
- **Cost Efficiency:** $0.000205 per query

