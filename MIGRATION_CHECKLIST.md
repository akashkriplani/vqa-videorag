# Migration & Testing Checklist

## ✅ Pre-Migration Checklist

- [ ] **Backup existing embeddings** (if any)

  ```bash
  cp -r faiss_db_openai_whisper_tiny/ faiss_db_backup_$(date +%Y%m%d)/
  cp -r feature_extraction_openai_whisper_tiny/ feature_extraction_backup_$(date +%Y%m%d)/
  ```

- [ ] **Review configuration files**

  - Check `.env` for HF_TOKEN
  - Verify model IDs in pipeline scripts
  - Confirm video directories exist

- [ ] **Check dependencies**
  ```bash
  pip list | grep -E "torch|transformers|faiss|open-clip|spacy|opencv"
  ```

---

## 🔄 Migration Steps

### Step 1: Regenerate Embeddings

- [ ] **Clear old embeddings** (optional, for clean start)

  ```bash
  rm -rf faiss_db_openai_whisper_tiny/*.index
  rm -rf faiss_db_openai_whisper_tiny/*.meta.json
  ```

- [ ] **Run embedding generation** with new pipeline

  ```bash
  python multimodal_pipeline_with_sliding_window.py
  ```

- [ ] **Verify segment_id presence** in output JSONs
  ```bash
  # Check a sample JSON file
  python -c "import json; data = json.load(open('feature_extraction_openai_whisper_tiny/textual/train/ABC123.json')); print('segment_id' in data[0])"
  ```

### Step 2: Test Query System

- [ ] **Test segment-level search**

  ```bash
  python query_faiss.py \
      --query "surgical procedure" \
      --text_index faiss_db_openai_whisper_tiny/textual_train.index \
      --visual_index faiss_db_openai_whisper_tiny/visual_train.index \
      --final_k 10 \
      --mode segment
  ```

- [ ] **Verify multimodal results** in output

  - Check for `🔗 [TEXT+VISUAL]` indicators
  - Confirm segment_id in results JSON
  - Verify combined scores are calculated

- [ ] **Test video-level search**

  ```bash
  python query_faiss.py \
      --query "medical diagnosis" \
      --mode video \
      --top_videos 5
  ```

- [ ] **Review output files**
  - `multimodal_search_results.json` should contain segment_id
  - Results should show text + visual evidence together

---

## 🧪 Validation Tests

### Test 1: Token Efficiency

- [ ] **Count embeddings** per video
  ```bash
  python -c "
  import json
  import glob
  files = glob.glob('feature_extraction_openai_whisper_tiny/textual/train/*.json')
  counts = [len(json.load(open(f))) for f in files[:10]]
  print(f'Avg embeddings per video: {sum(counts)/len(counts):.1f}')
  print(f'Expected: ~80-100')
  "
  ```
  - ✅ Should be ~80-100 embeddings/video (down from 150-200)

### Test 2: Multimodal Linking

- [ ] **Check segment_id consistency**
  ```bash
  python -c "
  import json
  text = json.load(open('feature_extraction_openai_whisper_tiny/textual/train/ABC123.json'))
  visual = json.load(open('feature_extraction_openai_whisper_tiny/visual/train/ABC123.json'))
  text_ids = {t['segment_id'] for t in text}
  visual_ids = {v['segment_id'] for v in visual}
  print(f'Text segments: {len(text_ids)}')
  print(f'Visual segments: {len(visual_ids)}')
  print(f'Overlap: {len(text_ids & visual_ids)}')
  print(f'Match rate: {len(text_ids & visual_ids) / max(len(text_ids), len(visual_ids)) * 100:.1f}%')
  "
  ```
  - ✅ Match rate should be ~90-100%

### Test 3: Deduplication

- [ ] **Check for content_hash in embeddings**
  ```bash
  python -c "
  import json
  data = json.load(open('feature_extraction_openai_whisper_tiny/textual/train/ABC123.json'))
  hashes = [d['content_hash'] for d in data]
  print(f'Total embeddings: {len(hashes)}')
  print(f'Unique hashes: {len(set(hashes))}')
  print(f'Duplicates: {len(hashes) - len(set(hashes))}')
  "
  ```
  - ✅ Should have 0 duplicates (deduplication integrated)

### Test 4: Score Distribution

- [ ] **Analyze query score ranges**
  ```bash
  python query_faiss.py --query "test query" --final_k 20 --mode segment
  # Check multimodal_search_results.json
  python -c "
  import json
  results = json.load(open('multimodal_search_results.json'))
  scores = [r['combined_score'] for r in results['results']]
  print(f'Score range: {min(scores):.3f} - {max(scores):.3f}')
  print(f'Expected: wider distribution than before (0.1-0.9)')
  "
  ```
  - ✅ Should have good score separation

---

## 🐛 Troubleshooting

### Issue: "KeyError: 'segment_id'"

**Diagnosis**: Using old embeddings without segment_id

```bash
python -c "
import json
sample = json.load(open('feature_extraction_openai_whisper_tiny/textual/train/ABC123.json'))[0]
print('Available keys:', list(sample.keys()))
"
```

**Solution**: Regenerate embeddings with new pipeline

### Issue: No multimodal matches

**Diagnosis**: Text and visual indices misaligned

```bash
ls faiss_db_openai_whisper_tiny/*.index
# Should see matching pairs: textual_train.index + visual_train.index
```

**Solution**: Ensure both indices generated from same video set

### Issue: Low combined scores

**Diagnosis**: Embeddings not normalized

```bash
python -c "
import json
import numpy as np
data = json.load(open('feature_extraction_openai_whisper_tiny/textual/train/ABC123.json'))
emb = np.array(data[0]['embedding'])
norm = np.linalg.norm(emb)
print(f'Embedding norm: {norm:.3f}')
print(f'Expected: ~1.0 (normalized)')
"
```

**Solution**: Check embedding_storage.py FaissDB.add() normalizes embeddings

### Issue: Out of memory

**Diagnosis**: local_k too high

```bash
# Reduce local_k parameter
python query_faiss.py --query "test" --local_k 30 --final_k 10
```

---

## 📊 Performance Benchmarks

### Expected Results (per video):

- **Embeddings Generated**: 80-100 (down from 150-200)
- **Processing Time**: ~28 seconds (down from ~45 seconds)
- **Memory Usage**: ~2-3 GB (down from ~4-5 GB)
- **Query Latency**: ~1-2 seconds for top-10 results

### Quality Metrics:

- **Multimodal Match Rate**: >90% (up from ~60%)
- **Score Separation**: 0.1-0.9 range (vs 0.4-0.6 before)
- **Deduplication Effectiveness**: 0 exact duplicates
- **Entity Extraction**: Same as before (if NER enabled)

---

## ✅ Final Verification

- [ ] **Embeddings regenerated** with segment_id
- [ ] **Query returns multimodal results** (text + visual paired)
- [ ] **Performance improved** (50% fewer embeddings)
- [ ] **Documentation reviewed** (REFACTORING_SUMMARY.md)
- [ ] **Backup created** (old embeddings saved)

---

## 📝 Sign-Off

**Migration Date**: **\*\***\_**\*\***

**Performed By**: **\*\***\_**\*\***

**Notes**:

```
- Embeddings per video: ________ (target: 80-100)
- Multimodal match rate: ________ (target: >90%)
- Query latency: ________ (target: <2s)
- Issues encountered: ________
```

---

## 🎉 Success Criteria

✅ All tests passed
✅ Multimodal results working
✅ Performance improved
✅ Documentation reviewed

**Status**: [ ] READY FOR PRODUCTION
