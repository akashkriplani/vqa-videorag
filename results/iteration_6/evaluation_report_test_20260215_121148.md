# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:26:32
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6176 | 0.4860 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6176 | 0.4860 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2235 | 0.2365 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.2766 | 0.3502 | 0.1339 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3637 | 0.3593 | 0.3244 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0904 | 0.0835 | 0.0749 |
| Temporal Precision | 0.2034 | 0.1988 | 0.1579 |
| Temporal Recall | 0.1537 | 0.1416 | 0.1395 |
| Temporal F1 | 0.1556 | 0.1338 | 0.1394 |

**Correct Video Retrieved Rate:** 92.16%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 99.41% | 0.0366 | 100.00% |
| Reduction Rate | 87.65% | 0.0424 | 90.00% |
| Avg Conflicts Detected | 6.64 | 6.45 | 5 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 33.37% | 0.1517 | 30.00% |
| Answer Length (words) | 140.5 | 60.4 | 168 |
| Tokens Used | 564.6 | 100.3 | 589 |
| Cost per Query | $0.000172 | $0.000053 | $0.000200 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0175

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.23 | 0.21 | 0.19 |
| Generation Time (s) | 8.22 | 2.47 | 8.72 |
| Total Time (s) | 8.45 | 2.46 | 8.90 |

**Total Evaluation Time:** 14.4 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.56% Temporal F1
- **Answer Quality:** 33.37% Confidence
- **Cost Efficiency:** $0.000172 per query

