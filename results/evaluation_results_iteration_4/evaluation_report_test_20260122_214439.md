# Evaluation Report: TEST Split

**Generated:** 2026-01-22 21:53:41
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6471 | 0.4779 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6471 | 0.4779 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1569 | 0.1361 | 0.2000 | 0.0000 | 0.6000 |
| mAP | 0.3961 | 0.4032 | 0.2500 | 0.0000 | 1.0000 |
| nDCG@10 | 0.4604 | 0.4019 | 0.4307 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0751 | 0.0703 | 0.0564 |
| Temporal Precision | 0.1834 | 0.1802 | 0.1261 |
| Temporal Recall | 0.1254 | 0.1230 | 0.0940 |
| Temporal F1 | 0.1321 | 0.1164 | 0.1068 |

**Correct Video Retrieved Rate:** 96.08%

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
| Confidence | 82.70% | 0.0598 | 84.82% |
| Answer Length (words) | 131.5 | 58.1 | 160 |
| Tokens Used | 747.1 | 87.4 | 793 |
| Cost per Query | $0.000198 | $0.000053 | $0.000224 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0202

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.23 | 0.13 | 0.17 |
| Generation Time (s) | 4.96 | 2.05 | 5.78 |
| Total Time (s) | 5.19 | 2.06 | 6.05 |

**Total Evaluation Time:** 8.8 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 13.21% Temporal F1
- **Answer Quality:** 82.70% Confidence
- **Cost Efficiency:** $0.000198 per query

