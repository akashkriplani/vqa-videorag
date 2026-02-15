# Evaluation Report: TEST Split

**Generated:** 2026-02-15 11:27:11
**Total Queries:** 102

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.3333 | 0.4714 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.3333 | 0.4714 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0725 | 0.1077 | 0.0000 | 0.0000 | 0.4000 |
| mAP | 0.1863 | 0.3397 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.2216 | 0.3548 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0687 | 0.0769 | 0.0454 |
| Temporal Precision | 0.1714 | 0.1970 | 0.1107 |
| Temporal Recall | 0.1072 | 0.1113 | 0.0776 |
| Temporal F1 | 0.1198 | 0.1234 | 0.0868 |

**Correct Video Retrieved Rate:** 54.90%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 86.76% | 0.0488 | 90.00% |
| Avg Conflicts Detected | 5.77 | 4.62 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 25.03% | 0.1542 | 30.00% |
| Answer Length (words) | 93.9 | 70.2 | 48 |
| Tokens Used | 443.0 | 218.0 | 493 |
| Cost per Query | $0.000119 | $0.000077 | $0.000090 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0121

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.24 | 0.22 | 0.16 |
| Generation Time (s) | 6.40 | 3.36 | 4.76 |
| Total Time (s) | 6.64 | 3.45 | 5.44 |

**Total Evaluation Time:** 11.3 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 11.98% Temporal F1
- **Answer Quality:** 25.03% Confidence
- **Cost Efficiency:** $0.000119 per query

