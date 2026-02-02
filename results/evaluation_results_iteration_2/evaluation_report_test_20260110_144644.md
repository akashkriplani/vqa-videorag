# Evaluation Report: TEST Split

**Generated:** 2026-01-10 15:00:14
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
| Pass Rate | 95.98% | 0.1323 | 100.00% |
| Reduction Rate | 85.49% | 0.0517 | 90.00% |
| Avg Conflicts Detected | 3.25 | 3.43 | 2 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 31.81% | 0.1522 | 30.00% |
| Answer Length (words) | 120.9 | 68.6 | 162 |
| Tokens Used | 560.0 | 115.5 | 584 |
| Cost per Query | $0.000158 | $0.000060 | $0.000194 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0161

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.16 | 0.13 | 0.15 |
| Generation Time (s) | 7.55 | 2.61 | 8.43 |
| Total Time (s) | 7.71 | 2.60 | 8.59 |

**Total Evaluation Time:** 13.1 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 13.21% Temporal F1
- **Answer Quality:** 31.81% Confidence
- **Cost Efficiency:** $0.000158 per query

