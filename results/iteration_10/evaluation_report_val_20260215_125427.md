# Evaluation Report: VAL Split

**Generated:** 2026-02-15 12:59:22
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6170 | 0.4861 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6170 | 0.4861 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2362 | 0.2440 | 0.2000 | 0.0000 | 1.2000 |
| mAP | 0.3100 | 0.3550 | 0.1670 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3939 | 0.3706 | 0.3743 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0844 | 0.0724 | 0.0853 |
| Temporal Precision | 0.1960 | 0.1979 | 0.1451 |
| Temporal Recall | 0.1559 | 0.1448 | 0.1242 |
| Temporal F1 | 0.1478 | 0.1198 | 0.1572 |

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
| Confidence | 46.29% | 0.2933 | 56.91% |
| Answer Length (words) | 103.1 | 59.0 | 137 |
| Tokens Used | 553.5 | 333.9 | 766 |
| Cost per Query | $0.000146 | $0.000098 | $0.000212 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0137

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.17 | 0.24 |
| Generation Time (s) | 2.72 | 2.14 | 3.59 |
| Total Time (s) | 3.00 | 2.18 | 3.83 |

**Total Evaluation Time:** 4.7 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 14.78% Temporal F1
- **Answer Quality:** 46.29% Confidence
- **Cost Efficiency:** $0.000146 per query

