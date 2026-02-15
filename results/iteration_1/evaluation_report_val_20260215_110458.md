# Evaluation Report: VAL Split

**Generated:** 2026-02-15 11:13:29
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.2660 | 0.4418 | 0.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.2660 | 0.4418 | 0.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.0617 | 0.1131 | 0.0000 | 0.0000 | 0.6000 |
| mAP | 0.1397 | 0.2975 | 0.0000 | 0.0000 | 1.0000 |
| nDCG@10 | 0.1706 | 0.3181 | 0.0000 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0592 | 0.0633 | 0.0414 |
| Temporal Precision | 0.1560 | 0.1844 | 0.1001 |
| Temporal Recall | 0.1061 | 0.1438 | 0.0618 |
| Temporal F1 | 0.1053 | 0.1080 | 0.0794 |

**Correct Video Retrieved Rate:** 60.64%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 100.00% | 0.0000 | 100.00% |
| Reduction Rate | 88.62% | 0.0345 | 90.00% |
| Avg Conflicts Detected | 3.96 | 2.77 | 3 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 22.34% | 0.1676 | 30.00% |
| Answer Length (words) | 69.5 | 62.5 | 31 |
| Tokens Used | 365.4 | 206.0 | 377 |
| Cost per Query | $0.000090 | $0.000070 | $0.000070 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0085

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.19 | 0.16 | 0.16 |
| Generation Time (s) | 5.01 | 2.64 | 3.91 |
| Total Time (s) | 5.19 | 2.66 | 4.07 |

**Total Evaluation Time:** 8.1 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 10.53% Temporal F1
- **Answer Quality:** 22.34% Confidence
- **Cost Efficiency:** $0.000090 per query

