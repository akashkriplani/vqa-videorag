# Evaluation Report: VAL Split

**Generated:** 2026-01-10 15:25:10
**Total Queries:** 94

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | 0.7128 | 0.4525 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10 | 0.7128 | 0.4525 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1745 | 0.1494 | 0.2000 | 0.0000 | 0.8000 |
| mAP | 0.4633 | 0.4164 | 0.3583 | 0.0000 | 1.0000 |
| nDCG@10 | 0.5266 | 0.4027 | 0.5254 | 0.0000 | 1.0000 |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | 0.0810 | 0.0808 | 0.0570 |
| Temporal Precision | 0.2004 | 0.2028 | 0.1245 |
| Temporal Recall | 0.1354 | 0.1399 | 0.0941 |
| Temporal F1 | 0.1403 | 0.1285 | 0.1079 |

**Correct Video Retrieved Rate:** 94.68%

---

## 2. Context Curation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Pass Rate | 98.19% | 0.1041 | 100.00% |
| Reduction Rate | 85.96% | 0.0512 | 90.00% |
| Avg Conflicts Detected | 3.95 | 3.09 | 4 |

---

## 3. Answer Generation Performance

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Confidence | 37.61% | 0.1488 | 30.00% |
| Answer Length (words) | 117.5 | 68.7 | 160 |
| Tokens Used | 550.2 | 128.9 | 589 |
| Cost per Query | $0.000156 | $0.000064 | $0.000196 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0147

---

## 4. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | 0.15 | 0.04 | 0.15 |
| Generation Time (s) | 8.01 | 2.96 | 9.24 |
| Total Time (s) | 8.17 | 2.97 | 9.40 |

**Total Evaluation Time:** 12.8 minutes

---

## Summary

- **Queries Processed:** 94
- **Average Performance:** 14.03% Temporal F1
- **Answer Quality:** 37.61% Confidence
- **Cost Efficiency:** $0.000156 per query

