# Evaluation Report: TEST Split

**Generated:** 2026-02-15 12:40:10
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.6373 | 0.4808 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.6373 | 0.4808 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.2176 | 0.2207 | 0.2000 | 0.0000 | 1.0000 |
| mAP | 0.2794 | 0.3454 | 0.1429 | 0.0000 | 1.0000 |
| nDCG@10 | 0.3712 | 0.3538 | 0.3333 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0869 | 0.0737 | 0.0770 |
| Temporal Precision | 0.2006 | 0.1838 | 0.1739 |
| Temporal Recall | 0.1466 | 0.1255 | 0.1329 |
| Temporal F1 | 0.1518 | 0.1201 | 0.1430 |

**Correct Video Retrieved Rate:** 91.18%

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
| Confidence | 48.51% | 0.2833 | 55.34% |
| Answer Length (words) | 105.0 | 65.3 | 148 |
| Tokens Used | 565.4 | 317.5 | 762 |
| Cost per Query | $0.000145 | $0.000094 | $0.000204 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0148

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.28 | 0.13 | 0.24 |
| Generation Time (s) | 2.41 | 1.88 | 3.32 |
| Total Time (s) | 2.69 | 1.88 | 3.66 |

**Total Evaluation Time:** 4.6 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 15.18% Temporal F1
- **Answer Quality:** 48.51% Confidence
- **Cost Efficiency:** $0.000145 per query

