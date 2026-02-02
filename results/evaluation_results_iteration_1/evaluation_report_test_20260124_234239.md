# Evaluation Report: TEST Split

**Generated:** 2026-01-24 23:56:46
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.3725 | 0.4835 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.3725 | 0.4835 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0882 | 0.1270 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.2298 | 0.3736 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.2649 | 0.3852 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0601 | 0.0655 | 0.0428 |
| Temporal Precision | 0.1561 | 0.1817 | 0.0995 |
| Temporal Recall | 0.0965 | 0.1021 | 0.0716 |
| Temporal F1 | 0.1067 | 0.1087 | 0.0821 |

**Correct Video Retrieved Rate:** 73.53%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 85.39% | 0.0554 | 90.00% |
| Avg Conflicts Detected | 4.90 | 4.27 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 28.54% | 0.1572 | 30.00% |
| Answer Length (words) | 114.8 | 70.9 | 160 |
| Tokens Used | 552.3 | 112.4 | 572 |
| Cost per Query | $0.000154 | $0.000061 | $0.000191 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0157

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.16 | 0.13 | 0.14 |
| Generation Time (s) | 7.96 | 3.18 | 9.12 |
| Total Time (s) | 8.12 | 3.19 | 9.27 |

**Total Evaluation Time:** 13.8 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 10.67% Temporal F1
- **Answer Quality:** 28.54% Confidence
- **Cost Efficiency:** $0.000154 per query

