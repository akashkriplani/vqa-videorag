# Hybrid Search - Quick Start Guide

## What is Hybrid Search?

Hybrid search combines two complementary retrieval methods:
1. **Dense Embeddings** (70%): Semantic similarity using neural models
2. **BM25** (30%): Lexical keyword matching with medical term expansion

## Why Use It?

**Problem**: Dense embeddings sometimes miss exact medical terminology
- Query: "mouth cancer check" 
- Dense model might not prioritize "oral cancer examination"

**Solution**: BM25 captures exact medical terms while dense models handle semantics
- Result: **Better ranking of expected videos** 📈

## Quick Start

### 1. Simple Hybrid Search
```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --hybrid \
  --text_index faiss_db/textual_train.index \
  --visual_index faiss_db/visual_train.index

# Note: --json_dir defaults to 'feature_extraction/textual'
# It automatically searches train/test/val subdirectories recursively
```

### 2. Compare with Pure Dense Search
```bash
# Automated comparison showing improvement
python compare_search_methods.py \
  --query "How to do a mouth cancer check at home?" \
  --expected_video "_6csIJAWj_s"
```

### 3. Run Quick Test
```bash
./test_hybrid.sh
```

## Key Parameters

### `--alpha` (Dense Weight)
Controls balance between semantic and keyword matching:

| Alpha | Dense | BM25 | Best For |
|-------|-------|------|----------|
| 0.9   | 90%   | 10%  | Abstract concepts, general queries |
| 0.7   | 70%   | 30%  | **Balanced (default, recommended)** |
| 0.5   | 50%   | 50%  | Medical terminology-heavy queries |
| 0.3   | 30%   | 70%  | Exact term matching critical |

**Example**:
```bash
# More semantic understanding
python query_faiss.py --query "..." --hybrid --alpha 0.8

# More keyword matching  
python query_faiss.py --query "..." --hybrid --alpha 0.5
```

### `--fusion` (Score Combination)
How to combine BM25 and dense scores:

- **linear** (default): `Combined = 0.7 × Dense + 0.3 × BM25`
  - Fast, intuitive, works well for most cases

- **rrf** (Reciprocal Rank Fusion): `Combined = Σ(1 / (60 + rank))`
  - More robust to score distribution differences
  - Good when dense and BM25 scales differ significantly

**Example**:
```bash
python query_faiss.py --query "..." --hybrid --fusion rrf
```

### `--expand_query`
Automatically expands medical terms with synonyms:
- "mouth cancer" → "oral cancer, oral cavity cancer, oropharyngeal cancer"
- "check" → "examination, screening, inspection, assessment"

**Enabled by default**. To disable:
```bash
python query_faiss.py --query "..." --hybrid --no-expand_query
```

### `--analyze_fusion`
Shows detailed contribution breakdown:
```bash
python query_faiss.py --query "..." --hybrid --analyze_fusion
```

Output:
```
1. _6csIJAWj_s_seg_1
   Combined: 0.7934
   Dense contribution: 0.5069 (70%)
   BM25 contribution: 0.2865 (30%)
   → BM25-driven result  # <-- BM25 found the right video!
```

## Example Workflows

### Workflow 1: Basic Query
```bash
# Your query (searches all splits by default)
python query_faiss.py \
  --query "How to treat a wound?" \
  --hybrid \
  --text_index faiss_db/textual_train.index

# Or specify a single split
python query_faiss.py \
  --query "How to treat a wound?" \
  --hybrid \
  --text_index faiss_db/textual_train.index \
  --json_dir feature_extraction/textual/train
```

### Workflow 2: Find Optimal Alpha
```bash
# Test different alpha values
for alpha in 0.5 0.6 0.7 0.8 0.9; do
  echo "Testing alpha=$alpha"
  python query_faiss.py \
    --query "Your query" \
    --hybrid \
    --alpha $alpha \
    --final_k 5
done
```

### Workflow 3: Medical Terminology Query
```bash
# Queries with specific medical terms benefit from lower alpha
python query_faiss.py \
  --query "diabetic ketoacidosis diagnosis" \
  --hybrid \
  --alpha 0.5 \  # Give more weight to BM25 for exact term matching
  --expand_query
```

### Workflow 4: Evaluate Multiple Queries
```bash
# Create evaluation script
queries=(
  "How to do a mouth cancer check at home?|_6csIJAWj_s"
  "How to apply a tourniquet?|_6VFqX5i4M8"
  "TMJ dislocation reduction|_9Vy1QsZ7k0"
)

for query_pair in "${queries[@]}"; do
  IFS='|' read -r query expected <<< "$query_pair"
  python compare_search_methods.py \
    --query "$query" \
    --expected_video "$expected"
done
```

## Results Interpretation

### Good Indicators
- ✅ Expected video in top 3 results
- ✅ "BM25-driven result" tags on medical terminology matches
- ✅ Combined scores > 0.5 for relevant results
- ✅ Query expansion finding synonym matches

### Signs to Tune
- ⚠️ Expected video beyond rank 5 → Lower alpha (more BM25)
- ⚠️ All results "Dense-driven" → Check query expansion, try RRF
- ⚠️ Very low BM25 scores → Add medical terms to expansion dictionary

## Customization

### Add Medical Terms
Edit `hybrid_search.py`, function `expand_medical_query()`:

```python
medical_synonyms = {
    "your term": ["synonym1", "synonym2", "synonym3"],
    # Add more terms here
}
```

Example additions:
```python
"diabetes": ["diabetic", "blood sugar", "glucose", "hyperglycemia"],
"asthma": ["bronchial asthma", "airway obstruction", "wheezing"],
"fracture": ["broken bone", "bone break", "skeletal injury"]
```

### Adjust BM25 Parameters
In `hybrid_search.py`, `HybridSearchEngine.__init__()`:

```python
self.bm25 = BM25Okapi(
    self.tokenized_corpus, 
    k1=1.5,  # Term frequency saturation (higher = more weight to repeated terms)
    b=0.75   # Length normalization (0 = no norm, 1 = full norm)
)
```

## Troubleshooting

### Issue: "No segments loaded"
**Solution**: Check `--json_dir` path points to correct feature extraction directory
```bash
ls feature_extraction/textual/train/  # Should show .json files
```

### Issue: "Hybrid search failed"
**Solution**: Ensure rank-bm25 is installed
```bash
pip install rank-bm25
```

### Issue: Poor results with hybrid search
**Try**:
1. Adjust alpha: `--alpha 0.5` (more BM25) or `--alpha 0.8` (more dense)
2. Try RRF fusion: `--fusion rrf`
3. Check query expansion is working: `--analyze_fusion`
4. Add domain-specific terms to expansion dictionary

## Performance Tips

1. **For medical terminology**: Use lower alpha (0.4-0.6)
2. **For conceptual queries**: Use higher alpha (0.7-0.9)
3. **When unsure**: Start with alpha=0.7 (default)
4. **View detailed analysis**: Always use `--analyze_fusion` during tuning

## Comparison Summary

From test on "mouth cancer check" query:

| Method | Rank | Score | Notes |
|--------|------|-------|-------|
| Dense Only | #2 | 0.5052 | Good semantic match |
| **Hybrid (α=0.7)** | **#1** | **0.5467** | **BM25 caught medical terms** ✅ |

**Improvement**: +1 position, moving expected video to #1

## Next Steps

After successful hybrid search implementation:

1. **Test on more queries** across different medical domains
2. **Expand medical dictionary** with domain-specific terms
3. **Implement cross-encoder re-ranking** (Step 2) for top-10 results
4. **Optimize alpha per query type** (automatic classification)
5. **Integrate with full evaluation pipeline**

## Support

- Full documentation: See `HYBRID_SEARCH_SUMMARY.md`
- Code: `hybrid_search.py`, `query_faiss.py`
- Tests: `test_hybrid.sh`, `compare_search_methods.py`

## Quick Reference

```bash
# Most common usage
python query_faiss.py --query "YOUR_QUERY" --hybrid

# With custom alpha
python query_faiss.py --query "YOUR_QUERY" --hybrid --alpha 0.6

# Full analysis
python query_faiss.py --query "YOUR_QUERY" --hybrid --analyze_fusion

# Compare methods
python compare_search_methods.py --query "YOUR_QUERY" --expected_video "VIDEO_ID"
```
