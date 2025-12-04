"""
multimodal_pipeline.py
Modular pipeline for multimodal medical video processing:
- ASR (openai/whisper-tiny)
- Entity recognition & text embeddings (BioBERT/ClinicalBERT, SciSpacy/HF NER)
- Frame extraction & visual embeddings (BiomedCLIP)
- FAISS DB integration
- Demo pipeline
"""
import os
import gc
import torch
import numpy as np
import open_clip
import getpass
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from video_processing import (
    transcribe_with_asr,
    load_ner_and_embed_models,
    extract_entities_and_embed,
    extract_frames_and_embed,
    deduplicate_embeddings_similarity,
    VideoProcessor
)

from video_processing.pipeline import VideoProcessorConfig

from embedding_storage import FaissDB, save_video_features, save_faiss_indices_from_lists
from data_preparation import filter_json_by_embeddings
from hierarchical_search_utils import hierarchical_search

# Import aggregation and output formatting from query_faiss for consistency
try:
    from query_faiss import aggregate_results_by_segment, print_segment_results, format_timestamp
except ImportError:
    print("Warning: Could not import from query_faiss. Multimodal aggregation may not be available.")
    aggregate_results_by_segment = None
    print_segment_results = None
    format_timestamp = None

# Import hybrid search functionality
try:
    from hybrid_search import HybridSearchEngine, load_segments_from_json_dir
except ImportError:
    print("Warning: Could not import hybrid_search. Hybrid search will not be available.")
    HybridSearchEngine = None
    load_segments_from_json_dir = None

# Parallel processing flag
ENABLE_PARALLEL = True  # Set to False to disable parallel processing

# Flag to use Colab secrets for HF_TOKEN (default: True)
USE_COLAB_SECRETS = True

# Load HuggingFace token from .env, Colab secrets, or prompt
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# if HF_TOKEN is None and USE_COLAB_SECRETS:
#     try:
#         HF_TOKEN = userdata.get('HF_TOKEN')
#     except Exception:
#         HF_TOKEN = None

if HF_TOKEN is None:
    try:
        HF_TOKEN = getpass.getpass("Enter your HuggingFace token (for gated models): ")
        hf_login(HF_TOKEN)
    except Exception as e:
        print("Warning: Could not login to HuggingFace. If you get a 401 error, set HF_TOKEN env variable or login manually.")
else:
    hf_login(HF_TOKEN)

def process_video_batch(batch_fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id,
                       use_new_processor=True, **processor_kwargs):
    """
    Process a batch of videos using either the new VideoProcessor or legacy implementation.

    Args:
        batch_fnames: List of video filenames to process
        video_dir: Directory containing videos
        text_feat_dir: Directory to save text features
        visual_feat_dir: Directory to save visual features
        asr_model_id: ASR model identifier
        use_new_processor: If True, use refactored VideoProcessor (recommended)
        **processor_kwargs: Additional configuration parameters for VideoProcessor

    Returns:
        Dictionary mapping filenames to (text_data, visual_data, error) tuples
    """
    batch_results = {}

    # NEW: Initialize VideoProcessor once for the batch if using new implementation
    if use_new_processor:
        from video_processing.pipeline import VideoProcessorConfig

        config = VideoProcessorConfig(
            asr_model_id=asr_model_id,
            window_size=processor_kwargs.get('window_size', 256),
            stride=processor_kwargs.get('stride', 192),
            min_coverage_contribution=processor_kwargs.get('min_coverage_contribution', 0.05),
            deduplication_mode=processor_kwargs.get('deduplication_mode', 'coverage'),
            frames_per_segment=processor_kwargs.get('frames_per_segment', 2),
            sampling_strategy=processor_kwargs.get('sampling_strategy', 'adaptive'),
            quality_filter=processor_kwargs.get('quality_filter', False),
            aggregation_method=processor_kwargs.get('aggregation_method', 'mean'),
            visual_similarity_threshold=processor_kwargs.get('visual_similarity_threshold', 0.98)
        )

        processor = VideoProcessor(config)

    for fname in batch_fnames:
        if not fname.endswith('.mp4'):
            batch_results[fname] = (None, None, 'Not an mp4 file')
            continue

        video_id = os.path.splitext(fname)[0]
        video_path = os.path.join(video_dir, fname)

        try:
            # Clear GPU memory before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            if use_new_processor:
                # NEW: Use refactored VideoProcessor
                print(f"\nProcessing {fname} with VideoProcessor...")
                (text_embs, text_meta), (visual_embs, visual_meta) = processor.process_video(
                    video_path=video_path,
                    video_id=video_id,
                    text_feat_dir=text_feat_dir,
                    visual_feat_dir=visual_feat_dir,
                    skip_if_exists=True
                )
                batch_results[fname] = ((text_embs, text_meta), (visual_embs, visual_meta), None)
            else:
                # LEGACY: Original inline implementation
                text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
                visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")

                # If BOTH JSON feature files already exist, load embeddings + metadata and skip processing
                if os.path.exists(text_json_path) and os.path.exists(visual_json_path):
                    try:
                        print(f"[SKIP] {fname}: Both text and visual JSONs exist. Loading from files...")
                        with open(text_json_path, 'r') as f:
                            text_json = json.load(f)
                        with open(visual_json_path, 'r') as f:
                            visual_json = json.load(f)

                        # Reconstruct embeddings and metadata
                        text_embs = [np.array(r['embedding']) if isinstance(r.get('embedding'), list) else np.array(r.get('embedding')) for r in text_json]
                        text_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in text_json]
                        visual_embs = [np.array(r['embedding']) if isinstance(r.get('embedding'), list) else np.array(r.get('embedding')) for r in visual_json]
                        visual_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in visual_json]

                        print(f"Loaded {len(text_embs)} text and {len(visual_embs)} visual embeddings from existing files.")
                        batch_results[fname] = ((text_embs, text_meta), (visual_embs, visual_meta), None)
                        continue  # Skip ASR and frame extraction
                    except Exception as e:
                        # If existing JSONs are corrupt/unreadable, reprocess the video
                        print(f"Warning: Failed to load existing JSONs for {fname}: {e}. Reprocessing video...")

                print(f"\nProcessing {fname}...")
                transcript_chunks = transcribe_with_asr(video_path, asr_model_id)
                print(f"Transcription complete: {len(transcript_chunks)} chunks extracted.")

                nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()

                # Use enhanced extraction with coverage-based deduplication
                print("Generating text embeddings with coverage-based deduplication...")
                text_results = extract_entities_and_embed(
                    transcript_chunks, nlp, bert_tokenizer, bert_model, video_id=video_id,
                    window_size=256, stride=192,
                    deduplication_mode='coverage',
                    min_coverage_contribution=0.05
                )
                print(f"Text embeddings generated: {len(text_results)} segments")

                visual_results = extract_frames_and_embed(video_path, text_results, video_id=video_id)
                print(f"Generated visual embeddings: {len(visual_results)} items.")

                # Apply similarity-based deduplication to visual embeddings
                print("Applying similarity-based deduplication to visual embeddings...")
                visual_results = deduplicate_embeddings_similarity(visual_results, similarity_threshold=0.98)
                print(f"Visual embeddings after deduplication: {len(visual_results)}")

                # Delegate saving to FAISS storage helper
                try:
                    save_video_features(video_id, text_results, visual_results, text_feat_dir, visual_feat_dir)
                except Exception:
                    pass

                # Enhanced logging
                print(f"\nResults Summary for {video_id}:")
                print(f"  - Text embeddings: {len(text_results)}")
                print(f"  - Visual embeddings: {len(visual_results)}")
                print(f"  - Text dim: {text_results[0]['embedding'].shape[0] if text_results else 'N/A'}")
                print(f"  - Visual dim: {visual_results[0]['embedding'].shape[0] if visual_results else 'N/A'}")

                text_embs = [r['embedding'] for r in text_results]
                text_meta = [{"video_id": video_id, **r} for r in text_results]
                visual_embs = [r['embedding'] for r in visual_results]
                visual_meta = [{"video_id": video_id, **r} for r in visual_results]
                batch_results[fname] = ((text_embs, text_meta), (visual_embs, visual_meta), None)

        except Exception as e:
            batch_results[fname] = (None, None, str(e))

    return batch_results

def parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id="openai/whisper-tiny", batch_size=1, max_workers=2):
    """
    Full pipeline parallel processing with batching: ASR, NER, embedding, JSON saving.
    """
    # Split fnames into batches
    batches = [fnames[i:i+batch_size] for i in range(0, len(fnames), batch_size)]
    results = {}
    total = len(fnames)
    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video_batch, batch, video_dir, text_feat_dir, visual_feat_dir, asr_model_id): tuple(batch) for batch in batches}
        for fut in as_completed(futures):
            batch_result = fut.result()
            for fname, (text, visual, error) in batch_result.items():
                processed += 1
                if error == 'JSONs already exist':
                    print(f"[SKIP] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                elif error:
                    print(f"[ERROR] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                else:
                    print(f"[DONE] {fname} | Progress: {processed}/{total} videos processed.")
                results[fname] = (text, visual)
    return results

# 4. Demo pipeline

# Hierarchical search functions now imported from hierarchical_search_utils
# Legacy wrapper for backward compatibility
def hierarchical_search_legacy(query_emb, text_db, json_data, top_k=5, enable_fine_grained=True):
    """
    Legacy wrapper for hierarchical_search - delegates to shared utility.
    Kept for backward compatibility with existing code.
    """
    return hierarchical_search(
        query_emb=query_emb,
        faiss_db=text_db,
        segments_data=json_data,
        top_k=top_k,
        enable_fine_grained=enable_fine_grained,
        result_format='multimodal'
    )

def demo_pipeline(video_path, text_feat_dir, visual_feat_dir, faiss_text_path, faiss_visual_path,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean'):
    """
    Demo pipeline with configurable hyperparameters.

    Text embedding hyperparameters:
        window_size: Token window size (default: 256)
        stride: Stride between windows (default: 192)
        min_coverage_contribution: Minimum new token coverage to keep window (default: 0.05)
        deduplication_mode: 'coverage', 'similarity', 'aggressive', or 'none' (default: 'coverage')

    Visual embedding hyperparameters:
        frames_per_segment: Number of frames per segment (default: 2)
        sampling_strategy: 'uniform', 'adaptive', or 'quality_based' (default: 'adaptive')
        quality_filter: Enable frame quality filtering (default: False)
        aggregation_method: 'mean' or 'max' (default: 'mean')
    """
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)

    fname = os.path.basename(video_path)
    video_dir = os.path.dirname(video_path)

    if not fname.endswith('.mp4'):
        return None, None

    video_id = os.path.splitext(fname)[0]
    video_path = os.path.join(video_dir, fname)
    text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
    visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")

    print(f"Processing video: {video_path}")

    # ASR
    transcript_chunks = transcribe_with_asr(video_path)

    print(f"Transcription complete: {len(transcript_chunks)} chunks extracted.")

    # NER + text embedding with configurable parameters
    nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()
    print(f"\nGenerating text embeddings with hyperparameters:")
    print(f"  window_size={window_size}, stride={stride}")
    print(f"  min_coverage_contribution={min_coverage_contribution}, deduplication_mode={deduplication_mode}")
    text_results = extract_entities_and_embed(
        transcript_chunks, nlp, bert_tokenizer, bert_model, video_id=video_id,
        window_size=window_size,
        stride=stride,
        deduplication_mode=deduplication_mode,
        min_coverage_contribution=min_coverage_contribution
    )

    print(f"\nText embeddings generated: {len(text_results)} segments")

    # Convert embeddings to lists for JSON serialization
    # Save textual features to JSON
    text_results_serializable = []
    for r in text_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        text_results_serializable.append(r_copy)
    with open(text_json_path, 'w') as f:
        json.dump(text_results_serializable, f)

    # Visual embedding with configurable parameters
    print(f"\nGenerating visual embeddings with hyperparameters:")
    print(f"  frames_per_segment={frames_per_segment}, sampling_strategy={sampling_strategy}")
    print(f"  quality_filter={quality_filter}, aggregation_method={aggregation_method}")
    visual_results = extract_frames_and_embed(
        video_path, text_results, video_id=video_id,
        frames_per_segment=frames_per_segment,
        sampling_strategy=sampling_strategy,
        quality_filter=quality_filter,
        aggregation_method=aggregation_method
    )

    print(f"Generated visual embeddings: {len(visual_results)} items.")

    # Apply similarity-based deduplication to visual embeddings
    print("\nApplying similarity-based deduplication to visual embeddings...")
    visual_results = deduplicate_embeddings_similarity(visual_results, similarity_threshold=0.98)

    print(f"Visual embeddings after deduplication: {len(visual_results)}")
    # Convert embeddings to lists for JSON serialization
    # Save visual features to JSON
    visual_results_serializable = []
    for r in visual_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        visual_results_serializable.append(r_copy)
    with open(visual_json_path, 'w') as f:
        json.dump(visual_results_serializable, f)
    # Save to FAISS
    text_db = FaissDB(dim=text_results[0]['embedding'].shape[0], index_path=faiss_text_path)
    visual_db = FaissDB(dim=visual_results[0]['embedding'].shape[0], index_path=faiss_visual_path)
    text_db.add([r['embedding'] for r in text_results], text_results)
    visual_db.add([r['embedding'] for r in visual_results], visual_results)
    text_db.save()
    visual_db.save()
    print("Databases saved.")

    # Enhanced logging for demo results
    print(f"\nFinal Results Summary:")
    print(f"- Video ID: {video_id}")
    print(f"- Text embeddings after deduplication: {len(text_results)}")
    print(f"- Visual embeddings after deduplication: {len(visual_results)}")
    print(f"- Text FAISS index dimension: {text_results[0]['embedding'].shape[0] if text_results else 'N/A'}")
    print(f"- Visual FAISS index dimension: {visual_results[0]['embedding'].shape[0] if visual_results else 'N/A'}")

    # Query demo with multimodal search (matching query_faiss.py output format)
    query = "How to do a mouth cancer check at home?"

    # Generate query embeddings for both text and visual modalities
    inputs = bert_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use CLS token pooling for consistency with embedding generation
        query_emb_text = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # Normalize text embedding for FAISS search
    norm = np.linalg.norm(query_emb_text)
    if norm > 0:
        query_emb_text = query_emb_text / norm

    # For visual search, we need BiomedCLIP text encoder
    # Load BiomedCLIP for cross-modal text->visual search
    print("Loading BiomedCLIP for cross-modal search...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        pretrained=True
    )
    clip_model.eval()

    # Tokenize with proper context length for BiomedCLIP
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokens = tokenizer([query])
    with torch.no_grad():
        query_emb_visual = clip_model.encode_text(tokens).squeeze().cpu().numpy()

    # Normalize visual embedding
    norm = np.linalg.norm(query_emb_visual)
    if norm > 0:
        query_emb_visual = query_emb_visual / norm

    print(f"\n{'='*80}")
    print(f"DEMO QUERY: {query}")
    print(f"{'='*80}")

    # Perform multimodal search
    print("\n[1] Searching textual index...")
    text_search_results = text_db.search(query_emb_text, top_k=50)  # Get more for hybrid search

    print("[2] Searching visual index...")
    visual_search_results = visual_db.search(query_emb_visual, top_k=10)

    print(f"\nFound {len(text_search_results)} text results and {len(visual_search_results)} visual results")

    # Apply hybrid search to text results if available
    use_hybrid = True  # Set to False to disable hybrid search in demo
    if use_hybrid and HybridSearchEngine is not None and load_segments_from_json_dir is not None:
        print(f"\n{'='*80}")
        print("APPLYING HYBRID SEARCH (BM25 + Dense Embeddings)")
        print(f"{'='*80}")

        try:
            # Load segments from the text feature directory
            segments_data = load_segments_from_json_dir(text_feat_dir)

            # Initialize hybrid search engine
            hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=0.7  # 70% dense, 30% BM25
            )

            # Transform FAISS results to format expected by hybrid search
            text_pool = []
            for result in text_search_results:
                text_pool.append({
                    'raw_score': result.get('distance', float('inf')),
                    'meta': result.get('metadata', {}),
                    'source_index': faiss_text_path
                })

            # Perform hybrid search
            hybrid_text_results = hybrid_engine.hybrid_search(
                query=query,
                dense_results=text_pool,
                top_k=50,
                fusion='linear',
                expand_query=True
            )

            # Show fusion analysis
            hybrid_engine.analyze_fusion_contribution(hybrid_text_results, top_k=10)

            # Replace text results with hybrid results
            text_search_results = []
            for hybrid_result in hybrid_text_results:
                meta = hybrid_result.get('metadata', {})
                # Convert hybrid score back to distance format
                combined_score = hybrid_result.get('combined_score', 0.0)
                distance = -np.log(combined_score + 1e-10)

                text_search_results.append({
                    'metadata': meta,
                    'distance': distance
                })

            print(f"✅ Hybrid search applied: Re-ranked {len(text_search_results)} text results")

        except Exception as e:
            print(f"⚠️  Hybrid search failed: {e}")
            print("Falling back to dense-only search...")
    elif use_hybrid:
        print("\n⚠️  Hybrid search not available (module not imported)")

    # Transform results to match query_faiss format
    def transform_faiss_results(results, is_visual=False):
        """Transform FaissDB search results to query_faiss format"""
        transformed = []
        for result in results:
            meta = result.get('metadata', {})
            dist = result.get('distance', float('inf'))
            transformed.append({
                'meta': meta,
                'raw_score': dist,
                'distance': dist
            })
        return transformed

    text_results_formatted = transform_faiss_results(text_search_results, is_visual=False)
    visual_results_formatted = transform_faiss_results(visual_search_results, is_visual=True)

    # Apply segment-level aggregation (matching query_faiss.py)
    if aggregate_results_by_segment is not None:
        print("\n[3] Aggregating multimodal results by segment...")
        # Use feature_extraction/ directory for hierarchical search (contains overlapping_timestamps)
        hierarchical_json_dir = "feature_extraction/"
        segment_contexts = aggregate_results_by_segment(
            text_results=text_results_formatted,
            visual_results=visual_results_formatted,
            top_k=10,
            text_weight=0.6,
            visual_weight=0.4,
            enable_hierarchical=True,
            json_dir=hierarchical_json_dir
        )

        # Print and save results in the same format as query_faiss.py
        # Determine output filename based on hybrid search usage
        output_file = "multimodal_search_results_hybrid.json" if use_hybrid else "multimodal_search_results_dense.json"

        if print_segment_results is not None:
            print_segment_results(segment_contexts, query=query, output_file=output_file)
        else:
            print(f"\nFound {len(segment_contexts)} multimodal segments")
            for i, ctx in enumerate(segment_contexts[:3], 1):
                print(f"\nSegment {i}: {ctx['video_id']} | Score: {ctx['combined_score']:.4f}")
                print(f"  Text: {ctx['text_evidence']['text'][:150] if ctx['text_evidence'] else 'N/A'}...")
    else:
        print("\nWarning: Multimodal aggregation not available. Install query_faiss dependencies.")
        print("Showing individual search results instead...")

        # Fallback: show individual results
        print("\nTop Text Results:")
        for i, result in enumerate(text_search_results[:3], 1):
            meta = result.get('metadata', {})
            print(f"  {i}. {meta.get('segment_id', 'N/A')} - Score: {result.get('distance', 'N/A'):.4f}")

        print("\nTop Visual Results:")
        for i, result in enumerate(visual_search_results[:3], 1):
            meta = result.get('metadata', {})
            print(f"  {i}. {meta.get('segment_id', 'N/A')} - Score: {result.get('distance', 'N/A'):.4f}")

def process_video(fname, video_dir, text_feat_dir, visual_feat_dir,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean',
                 use_new_processor=True):
    """
    Process a single video with configurable hyperparameters.

    Args:
        use_new_processor: If True, use refactored VideoProcessor class (recommended).
                          If False, use legacy inline implementation.

    See demo_pipeline docstring for parameter descriptions.
    """
    if not fname.endswith('.mp4'):
        return None, None

    video_id = os.path.splitext(fname)[0]
    video_path = os.path.join(video_dir, fname)

    if use_new_processor:
        from video_processing.pipeline import VideoProcessorConfig

        config = VideoProcessorConfig(
            window_size=window_size,
            stride=stride,
            min_coverage_contribution=min_coverage_contribution,
            deduplication_mode=deduplication_mode,
            frames_per_segment=frames_per_segment,
            sampling_strategy=sampling_strategy,
            quality_filter=quality_filter,
            aggregation_method=aggregation_method
        )

        processor = VideoProcessor(config)
        return processor.process_video(
            video_path, video_id,
            text_feat_dir=text_feat_dir,
            visual_feat_dir=visual_feat_dir,
            skip_if_exists=True
        )

    # LEGACY: Original inline implementation (kept for backward compatibility)
    text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
    visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")
    # If both JSON files exist, load embeddings from them and return (SKIP ASR + FRAME EXTRACTION)
    if os.path.exists(text_json_path) and os.path.exists(visual_json_path):
        print(f"[SKIP] {fname}: Both text and visual JSONs exist. Loading from files (no ASR/frame extraction needed)...")
        try:
            with open(text_json_path, 'r') as f:
                text_json = json.load(f)
            with open(visual_json_path, 'r') as f:
                visual_json = json.load(f)

            # Reconstruct embeddings and metadata
            text_embs = [np.array(r['embedding']) for r in text_json]
            text_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in text_json]

            visual_embs = [np.array(r['embedding']) for r in visual_json]
            visual_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in visual_json]

            print(f"✅ Loaded {len(text_embs)} text and {len(visual_embs)} visual embeddings from existing files (ASR/frame extraction skipped).")
            return (text_embs, text_meta), (visual_embs, visual_meta)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load existing JSONs: {e}. Reprocessing video...")
            # Fall through to reprocess the video

    # Each thread loads its own models
    nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()

    try:
        print(f"\nProcessing {fname}...")
        transcript_chunks = transcribe_with_asr(video_path)
        print(f"Transcription complete: {len(transcript_chunks)} chunks extracted.")
    except Exception as e:
        print(f"ASR failed for {video_path}: {e}")
        return None, None

    # Use enhanced extraction with configurable parameters
    print(f"Generating text embeddings (ws={window_size}, stride={stride}, dedup={deduplication_mode})...")
    text_results = extract_entities_and_embed(
        transcript_chunks, nlp, bert_tokenizer, bert_model, video_id=video_id,
        window_size=window_size,
        stride=stride,
        deduplication_mode=deduplication_mode,
        min_coverage_contribution=min_coverage_contribution
    )
    print(f"Text embeddings generated: {len(text_results)} segments")

    # Convert embeddings to lists for JSON serialization
    text_results_serializable = []
    for r in text_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        text_results_serializable.append(r_copy)
    with open(text_json_path, 'w') as f:
        json.dump(text_results_serializable, f)
    text_embs = [r['embedding'] for r in text_results]
    text_meta = [{"video_id": video_id, **r} for r in text_results]

    visual_results = extract_frames_and_embed(
        video_path, text_results, video_id=video_id,
        frames_per_segment=frames_per_segment,
        sampling_strategy=sampling_strategy,
        quality_filter=quality_filter,
        aggregation_method=aggregation_method
    )
    print(f"Generated visual embeddings: {len(visual_results)} items.")

    # Apply similarity-based deduplication to visual embeddings
    print("Applying similarity-based deduplication to visual embeddings...")
    visual_results = deduplicate_embeddings_similarity(visual_results, similarity_threshold=0.98)
    print(f"Visual embeddings after deduplication: {len(visual_results)}")

    # Convert embeddings to lists for JSON serialization
    visual_results_serializable = []
    for r in visual_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        visual_results_serializable.append(r_copy)
    with open(visual_json_path, 'w') as f:
        json.dump(visual_results_serializable, f)
    visual_embs = [r['embedding'] for r in visual_results]
    visual_meta = [{"video_id": video_id, **r} for r in visual_results]

    # Enhanced logging
    print(f"\nResults Summary for {video_id}:")
    print(f"  - Text embeddings: {len(text_results)}")
    print(f"  - Visual embeddings: {len(visual_results)}")
    print(f"  - Text dim: {text_results[0]['embedding'].shape[0] if text_results else 'N/A'}")
    print(f"  - Visual dim: {visual_results[0]['embedding'].shape[0] if visual_results else 'N/A'}")

    return (text_embs, text_meta), (visual_embs, visual_meta)


def process_split(split, video_dir, text_feat_dir, visual_feat_dir, faiss_text_path, faiss_visual_path,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean'):
    """
    Process a full split (train/val/test) with configurable hyperparameters.

    See demo_pipeline docstring for parameter descriptions.
    """
    print(f"\nProcessing split '{split}' with hyperparameters:")
    print(f"  Text: window_size={window_size}, stride={stride}, min_cov={min_coverage_contribution}, dedup={deduplication_mode}")
    print(f"  Visual: frames={frames_per_segment}, strategy={sampling_strategy}, quality={quality_filter}, agg={aggregation_method}")

    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    all_text_embs, all_text_meta = [], []
    all_visual_embs, all_visual_meta = [], []
    fnames = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    batch_size = 4  # Tune for your hardware
    max_workers = 2 if torch.cuda.is_available() or torch.backends.mps.is_available() else os.cpu_count()
    if ENABLE_PARALLEL:
        print(f"Parallel processing enabled: batch_size={batch_size}, max_workers={max_workers}")
        results = parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, batch_size=batch_size, max_workers=max_workers)
        for fname in fnames:
            text, visual = results.get(fname, (None, None))
            if text and all(item is not None for item in text):
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual and all(item is not None for item in visual):
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    else:
        total = len(fnames)
        done = 0
        for idx, fname in enumerate(fnames, 1):
            text, visual = process_video(
                fname, video_dir, text_feat_dir, visual_feat_dir,
                window_size=window_size, stride=stride,
                min_coverage_contribution=min_coverage_contribution,
                deduplication_mode=deduplication_mode,
                frames_per_segment=frames_per_segment,
                sampling_strategy=sampling_strategy,
                quality_filter=quality_filter,
                aggregation_method=aggregation_method
            )
            if text:
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual:
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            done += 1
            print(f"Progress: {done}/{total} videos processed.")
    # Delegate FAISS index building / saving to embedding_storage
    try:
        save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path)
    except Exception:
        # Fallback: try previous logic if the helper fails
        if all_text_embs:
            print("Saving text embeddings to FAISS (fallback)...")
            text_db = FaissDB(dim=len(all_text_embs[0]), index_path=faiss_text_path)
            text_db.add(all_text_embs, all_text_meta)
            text_db.save()
        if all_visual_embs:
            print("Saving visual embeddings to FAISS (fallback)...")
            visual_db = FaissDB(dim=len(all_visual_embs[0]), index_path=faiss_visual_path)
            visual_db.add(all_visual_embs, all_visual_meta)
            visual_db.save()
    print(f"Done processing split: {split}")

def test_single_video(video_path, text_feat_dir, visual_feat_dir):
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    fname = os.path.basename(video_path)
    (text_embs, text_meta), (visual_embs, visual_meta) = process_video(fname, os.path.dirname(video_path), text_feat_dir, visual_feat_dir)
    print(f"Text embeddings: {len(text_embs)}; Visual embeddings: {len(visual_embs)}")
    print(f"Sample text meta: {text_meta[0] if text_meta else None}")
    print(f"Sample visual meta: {visual_meta[0] if visual_meta else None}")
    save_faiss_indices_from_lists(text_embs, text_meta, visual_embs, visual_meta,
                                  faiss_text_path='faiss_db/textual_single.index',
                                  faiss_visual_path='faiss_db/visual_single.index')

def main():
    # Test mode for a single video
    # test_single_video("videos_train/_6csIJAWj_s.mp4", "feature_extraction/textual/test_single", "feature_extraction/visual/test_single")

    # Test demo pipeline
    # demo_pipeline(
    #     video_path="videos_train/_6csIJAWj_s.mp4",
    #     text_feat_dir="feature_extraction/textual/demo",
    #     visual_feat_dir="feature_extraction/visual/demo",
    #     faiss_text_path="faiss_db/textual_demo.index",
    #     faiss_visual_path="faiss_db/visual_demo.index"
    # )

    # Uncomment above and set your video path to test single video or run a demo pipeline
    splits = [
        ("train", "videos_train"),
        ("val", "videos_val"),
        ("test", "videos_test")
    ]

    for split, video_dir in splits:
        print(f"Processing split: {split}")
        process_split(
            split=split,
            video_dir=video_dir,
            text_feat_dir=f"feature_extraction/textual/{split}",
            visual_feat_dir=f"feature_extraction/visual/{split}",
            faiss_text_path=f"faiss_db/textual_{split}.index",
            faiss_visual_path=f"faiss_db/visual_{split}.index"
        )

    # After all splits are processed, filter JSON files based on successfully generated embeddings
    print("\n" + "="*80)
    print("Filtering JSON files to keep only entries with both textual AND visual embeddings...")
    print("="*80)
    filter_json_by_embeddings(model_name="openai/whisper-tiny")

if __name__ == "__main__":
    main()