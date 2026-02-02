# Evaluation Report: TEST Split

**Generated:** 2026-01-25 23:22:11
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2882 | 0.2680 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.4015 | 0.3765 | 0.3250 | 0.0000 | 1.0000 |
| nDCG@10 | 0.4864 | 0.3828 | 0.5000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0930 | 0.0802 | 0.0761 |
| Temporal Precision | 0.2150 | 0.2077 | 0.1635 |
| Temporal Recall | 0.1517 | 0.1231 | 0.1368 |
| Temporal F1 | 0.1608 | 0.1296 | 0.1414 |

**Correct Video Retrieved Rate:** 97.06%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 96.47% | 0.1303 | 100.00% |
| Reduction Rate | 88.33% | 0.0373 | 90.00% |
| Avg Conflicts Detected | 5.92 | 5.80 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 31.30% | 0.1493 | 30.00% |
| Answer Length (words) | 131.8 | 66.6 | 168 |
| Tokens Used | 543.7 | 104.9 | 581 |
| Cost per Query | $0.000162 | $0.000057 | $0.000195 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0165

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.17 | 0.08 | 0.15 |
| Generation Time (s) | 8.02 | 2.78 | 9.12 |
| Total Time (s) | 8.19 | 2.77 | 9.34 |

**Total Evaluation Time:** 13.9 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 16.08% Temporal F1
- **Answer Quality:** 31.30% Confidence
- **Cost Efficiency:** $0.000162 per query

