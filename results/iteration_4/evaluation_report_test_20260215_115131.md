# Evaluation Report: TEST Split

**Generated:** 2026-02-15 11:56:39
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 57.59% | 0.2437 | 63.28% |
| Answer Length (words) | 106.1 | 66.4 | 151 |
| Tokens Used | 631.2 | 259.9 | 774 |
| Cost per Query | $0.000159 | $0.000084 | $0.000214 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0162

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.19 | 0.12 | 0.17 |
| Generation Time (s) | 2.71 | 1.95 | 3.33 |
| Total Time (s) | 2.90 | 1.97 | 3.48 |

**Total Evaluation Time:** 4.9 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 12.52% Temporal F1
- **Answer Quality:** 57.59% Confidence
- **Cost Efficiency:** $0.000159 per query

