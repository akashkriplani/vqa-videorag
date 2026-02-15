# Evaluation Report: VAL Split

**Generated:** 2026-02-15 11:39:37
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.3830 | 0.4861 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.3830 | 0.4861 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0915 | 0.1294 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.2808 | 0.4073 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3082 | 0.4185 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0710 | 0.0782 | 0.0483 |
| Temporal Precision | 0.1764 | 0.1949 | 0.1174 |
| Temporal Recall | 0.1178 | 0.1450 | 0.0829 |
| Temporal F1 | 0.1233 | 0.1271 | 0.0921 |

**Correct Video Retrieved Rate:** 71.28%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 86.70% | 0.0470 | 90.00% |
| Avg Conflicts Detected | 3.96 | 3.02 | 3 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 29.40% | 0.1978 | 30.00% |
| Answer Length (words) | 104.5 | 69.3 | 126 |
| Tokens Used | 462.3 | 212.6 | 532 |
| Cost per Query | $0.000129 | $0.000077 | $0.000161 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0122

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.57 | 0.77 | 0.22 |
| Generation Time (s) | 10.38 | 8.72 | 9.06 |
| Total Time (s) | 10.94 | 9.02 | 9.47 |

**Total Evaluation Time:** 17.1 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 12.33% Temporal F1
- **Answer Quality:** 29.40% Confidence
- **Cost Efficiency:** $0.000129 per query

