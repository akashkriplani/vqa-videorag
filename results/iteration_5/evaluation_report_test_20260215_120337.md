# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:08:55
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 48.97% | 0.2872 | 56.25% |
| Answer Length (words) | 102.3 | 62.3 | 144 |
| Tokens Used | 565.4 | 317.3 | 756 |
| Cost per Query | $0.000145 | $0.000094 | $0.000203 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0148

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.43 | 0.37 | 0.38 |
| Generation Time (s) | 2.55 | 2.01 | 3.34 |
| Total Time (s) | 2.99 | 2.03 | 3.58 |

**Total Evaluation Time:** 5.1 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.56% Temporal F1
- **Answer Quality:** 48.97% Confidence
- **Cost Efficiency:** $0.000145 per query

