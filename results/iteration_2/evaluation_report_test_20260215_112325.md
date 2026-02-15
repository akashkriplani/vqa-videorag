# Evaluation Report: TEST Split

**Generated:** 2026-02-15 11:44:36
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.4118 | 0.4922 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.4118 | 0.4922 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0922 | 0.1210 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.2710 | 0.3990 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3059 | 0.4079 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0720 | 0.0776 | 0.0510 |
| Temporal Precision | 0.1774 | 0.1971 | 0.1129 |
| Temporal Recall | 0.1157 | 0.1306 | 0.0860 |
| Temporal F1 | 0.1252 | 0.1254 | 0.0970 |

**Correct Video Retrieved Rate:** 76.47%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 99.90% | 0.0099 | 100.00% |
| Reduction Rate | 85.29% | 0.0499 | 90.00% |
| Avg Conflicts Detected | 5.60 | 4.78 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 27.80% | 0.1812 | 30.00% |
| Answer Length (words) | 107.8 | 70.2 | 154 |
| Tokens Used | 489.5 | 211.6 | 559 |
| Cost per Query | $0.000136 | $0.000077 | $0.000171 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0138

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.62 | 0.58 | 0.27 |
| Generation Time (s) | 11.60 | 9.17 | 9.84 |
| Total Time (s) | 12.22 | 9.52 | 10.00 |

**Total Evaluation Time:** 20.8 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 12.52% Temporal F1
- **Answer Quality:** 27.80% Confidence
- **Cost Efficiency:** $0.000136 per query

