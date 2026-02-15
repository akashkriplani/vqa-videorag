# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:52:18
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.5532 | 0.4972 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.5532 | 0.4972 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1915 | 0.2314 | 0.2000 | 0.0000 | 1.2000 |
| mAP | 0.2718 | 0.3496 | 0.1181 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3471 | 0.3699 | 0.3082 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0752 | 0.0660 | 0.0630 |
| Temporal Precision | 0.1885 | 0.2036 | 0.1301 |
| Temporal Recall | 0.1331 | 0.1274 | 0.1198 |
| Temporal F1 | 0.1331 | 0.1116 | 0.1186 |

**Correct Video Retrieved Rate:** 87.23%

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
| Confidence | 49.37% | 0.3006 | 63.44% |
| Answer Length (words) | 110.8 | 60.4 | 150 |
| Tokens Used | 559.9 | 336.8 | 782 |
| Cost per Query | $0.000149 | $0.000099 | $0.000214 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0140

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.30 | 0.18 | 0.24 |
| Generation Time (s) | 2.74 | 2.04 | 3.91 |
| Total Time (s) | 3.03 | 2.07 | 4.11 |

**Total Evaluation Time:** 4.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 13.31% Temporal F1
- **Answer Quality:** 49.37% Confidence
- **Cost Efficiency:** $0.000149 per query

