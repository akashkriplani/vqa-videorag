# Evaluation Report: TEST Split

**Generated:** 2026-01-28 16:43:12
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2882 | 0.2680 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.4059 | 0.3780 | 0.3250 | 0.0000 | 1.0000 |
| nDCG@10 | 0.4901 | 0.3848 | 0.5000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0924 | 0.0807 | 0.0780 |
| Temporal Precision | 0.2134 | 0.2070 | 0.1635 |
| Temporal Recall | 0.1501 | 0.1220 | 0.1368 |
| Temporal F1 | 0.1595 | 0.1304 | 0.1447 |

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
| Confidence | 81.70% | 0.0667 | 84.23% |
| Answer Length (words) | 144.6 | 49.6 | 163 |
| Tokens Used | 767.0 | 75.1 | 796 |
| Cost per Query | $0.000210 | $0.000045 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0214

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.35 | 0.16 | 0.36 |
| Generation Time (s) | 4.90 | 1.72 | 5.26 |
| Total Time (s) | 5.25 | 1.74 | 5.62 |

**Total Evaluation Time:** 8.9 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.95% Temporal F1
- **Answer Quality:** 81.70% Confidence
- **Cost Efficiency:** $0.000210 per query

