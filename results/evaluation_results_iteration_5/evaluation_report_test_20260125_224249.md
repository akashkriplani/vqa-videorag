# Evaluation Report: TEST Split

**Generated:** 2026-01-25 22:51:31
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 81.71% | 0.0667 | 84.26% |
| Answer Length (words) | 145.1 | 48.6 | 162 |
| Tokens Used | 768.3 | 72.2 | 796 |
| Cost per Query | $0.000210 | $0.000044 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0214

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.23 | 0.13 | 0.19 |
| Generation Time (s) | 4.76 | 1.58 | 5.12 |
| Total Time (s) | 5.00 | 1.58 | 5.31 |

**Total Evaluation Time:** 8.5 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 16.08% Temporal F1
- **Answer Quality:** 81.71% Confidence
- **Cost Efficiency:** $0.000210 per query

