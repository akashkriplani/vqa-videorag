# Evaluation Report: TEST Split

**Generated:** 2026-01-28 13:42:17
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6961 | 0.4599 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2902 | 0.2681 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.4065 | 0.3776 | 0.3250 | 0.0000 | 1.0000 |
| nDCG@10 | 0.4909 | 0.3845 | 0.5000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0934 | 0.0813 | 0.0780 |
| Temporal Precision | 0.2173 | 0.2114 | 0.1694 |
| Temporal Recall | 0.1504 | 0.1215 | 0.1361 |
| Temporal F1 | 0.1611 | 0.1312 | 0.1447 |

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
| Confidence | 81.73% | 0.0666 | 84.33% |
| Answer Length (words) | 143.9 | 49.4 | 162 |
| Tokens Used | 768.5 | 75.0 | 799 |
| Cost per Query | $0.000210 | $0.000045 | $0.000231 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0214

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.13 | 0.22 |
| Generation Time (s) | 5.49 | 1.96 | 5.74 |
| Total Time (s) | 5.77 | 1.97 | 6.10 |

**Total Evaluation Time:** 9.8 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 16.11% Temporal F1
- **Answer Quality:** 81.73% Confidence
- **Cost Efficiency:** $0.000210 per query

