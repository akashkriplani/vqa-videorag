# Evaluation Report: VAL Split

**Generated:** 2026-02-15 11:51:21
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
| Pass Rate | 0.00% | 0.0000 | 0.00% |
| Reduction Rate | 0.00% | 0.0000 | 0.00% |
| Avg Conflicts Detected | 0.00 | 0.00 | 0 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 58.34% | 0.2485 | 67.42% |
| Answer Length (words) | 116.6 | 64.1 | 152 |
| Tokens Used | 637.8 | 269.9 | 784 |
| Cost per Query | $0.000167 | $0.000085 | $0.000219 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0157

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.41 | 0.32 | 0.39 |
| Generation Time (s) | 2.88 | 1.82 | 3.62 |
| Total Time (s) | 3.29 | 1.90 | 3.97 |

**Total Evaluation Time:** 5.2 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 12.33% Temporal F1
- **Answer Quality:** 58.34% Confidence
- **Cost Efficiency:** $0.000167 per query

