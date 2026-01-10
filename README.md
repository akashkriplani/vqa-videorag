# Enhancing Visual Question Answering Using Retrieval Augmented Generation in Medical Domain

A comprehensive Retrieval-Augmented Generation (RAG) system for answering medical questions from educational videos using state-of-the-art multimodal embeddings, hybrid search, and LLM-based answer generation.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Usage Guide](#-detailed-usage-guide)
- [Project Structure](#-project-structure)
- [Configuration Options](#-configuration-options)
- [Example Workflows](#-example-workflows)
- [Performance Benchmarks](#-performance-benchmarks)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This project implements a **Medical VideoRAG** system that processes medical educational videos to answer natural language questions. The system extracts both textual (speech) and visual information from videos, creates searchable embeddings, and generates accurate, evidence-based answers with proper source attribution.

**Key Innovation**: Combines dense semantic search (BiomedCLIP embeddings) with sparse lexical search (BM25) for optimal retrieval in the medical domain, enhanced with adaptive context curation and self-reflection attribution.

**Dataset**: Built on the [MedVidQA dataset](https://github.com/deepaknlp/MedVidQA) - 800+ medical YouTube videos with 3,000+ question-answer pairs.

## ✨ Key Features

### 🔍 Multimodal Processing

- **Automatic Speech Recognition**: OpenAI Whisper (tiny) for fast, accurate medical transcription
- **Text Embeddings**: BiomedCLIP text encoder with sliding window segmentation (256 tokens, 192 stride)
- **Visual Embeddings**: BiomedCLIP vision encoder with adaptive frame sampling
- **Medical NER**: SciSpacy entity recognition for medical terminology extraction

### 🔎 Advanced Search

- **Dense Search**: FAISS-based vector similarity (BiomedCLIP embeddings)
- **Sparse Search**: BM25 keyword matching with medical term expansion
- **Hybrid Fusion**: Configurable linear/RRF fusion (default: α=0.3, 70% BM25 + 30% dense)
- **Hierarchical Search**: Extended context retrieval with overlapping timestamps

### 🧠 Intelligent Answer Generation

- **Context Curation**: Multi-stage filtering with NLI factuality scoring
- **Query Classification**: Procedural, diagnostic, factoid, or general
- **Self-Reflection Attribution**: Claim-level evidence mapping
- **Cost-Optimized LLM**: GPT-4o-mini (~$0.0003-0.0005 per query)

### 📊 Comprehensive Evaluation

- **Retrieval Metrics**: Recall@K, MRR, temporal IoU/F1, video hit rate
- **Generation Metrics**: ROUGE-L, answer quality, confidence scores
- **Performance Timing**: End-to-end latency tracking
- **Cost Tracking**: Token usage and API cost estimation

## 🏗️ System Architecture

The system operates in **6 phases**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Understanding and Preprocessing                   │
│ • Download videos from YouTube                                  │
│ • Clean dataset (remove duplicates, validate timestamps)        │
│ • Split into train/val/test                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Multimodal Feature Extraction                          │
│ • ASR: Extract audio → Whisper → Transcripts with timestamps    │
│ • Text: Sliding windows → Medical NER → BiomedCLIP embeddings   │
│ • Visual: Adaptive sampling → Frame extraction → CLIP embeddings│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Embedding Storage & Indexing                           │
│ • Normalize embeddings (L2 norm)                                │
│ • Build FAISS indices (separate text/visual, train/val/test)    │
│ • Persist metadata (JSON) with segment info & timestamps        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Multimodal Search and Retrieval                        │
│ • Query embedding (BiomedCLIP)                                  │
│ • Parallel search: Dense (FAISS) + Sparse (BM25)                │
│ • Hybrid fusion (linear/RRF)                                    │
│ • Multimodal aggregation (text + visual by segment_id)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Answer Generation with Attribution                     │
│ • Context curation (quality filtering, NLI scoring)             │
│ • Query classification (procedural/diagnostic/factoid/general)  │
│ • GPT-4o-mini generation (temperature=0.3, max_tokens=250)      │
│ • Self-reflection attribution (claim→evidence mapping)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: Evaluation and Experimentation                         │
│ • Retrieval metrics: Recall@K, MRR, temporal IoU/F1             │
│ • Generation metrics: Answer quality, confidence                │
│ • Ablation studies: Dense vs hybrid, fusion strategies          │
│ • Hyperparameter optimization: Grid search, performance plots   │
│ • Cost analysis: Token usage, API costs, latency profiling      │
└─────────────────────────────────────────────────────────────────┘
```

![VideoRAG Architecture for Medical VQA](images/System_Architecture.png)

## 💻 Hardware Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB free (videos: ~30GB, features: ~10GB, indices: ~5GB)
- **GPU**: Optional (speeds up embedding generation 3-5x)

### Recommended Configuration

- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 32GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 Ti or better) OR Apple M1/M2 with 16GB+ unified memory
- **Storage**: SSD with 100GB+ free space

### Performance Notes

- **Without GPU**: Full pipeline on 899 videos takes ~24-48 hours
- **With GPU (CUDA/MPS)**: ~6-12 hours
- **Parallel Processing**: Adjust `max_workers` in [multimodal_pipeline_with_sliding_window.py](multimodal_pipeline_with_sliding_window.py) based on your CPU/GPU

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/akashkriplani/vqa-videorag.git
```

### 2. Create Python Environment

```bash
# Using conda (recommended)
conda create -n medvidrag python=3.8
conda activate medvidrag

# OR using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install System Dependencies

**macOS:**

```bash
brew install ffmpeg
npm install -g youtube-po-token-generator
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install ffmpeg
npm install -g youtube-po-token-generator
```

**Windows:**

- Download FFmpeg from https://ffmpeg.org/download.html
- Add to PATH
- Install Node.js and run: `npm install -g youtube-po-token-generator`

### 4. Install Python Dependencies

**Standard Installation:**

```bash
pip install -r requirements.txt
```

**Install SciSpacy Model:**

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

**Download NLTK Data:**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 5. GPU Setup (Optional but Recommended)

**For NVIDIA GPUs:**

```bash
# Uninstall CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu>=1.7.4
```

**For Apple Silicon (M1/M2/M3):**

- PyTorch automatically uses MPS (Metal Performance Shaders)
- No additional setup required

### 6. Configure Environment Variables

Create a `.env` file:

```bash
# OpenAI API key (required for answer generation)
OPENAI_API_KEY=your_openai_api_key_here

# HuggingFace token (for gated models like BiomedCLIP)
HF_TOKEN=your_huggingface_token_here
```

Get your API keys:

- OpenAI: https://platform.openai.com/api-keys
- HuggingFace: https://huggingface.co/settings/tokens

## 🎬 Quick Start

### Option 1: Full Pipeline (Recommended for First-Time Users)

```bash
# Run everything: data prep → feature extraction → indexing → demo query
python main.py
```

This executes:

1. [data_preparation.py](data_preparation.py) - Download videos, clean dataset
2. [eda_medvidqa.py](eda_medvidqa.py) - Exploratory data analysis
3. [multimodal_pipeline_with_sliding_window.py](multimodal_pipeline_with_sliding_window.py) - Extract features, build indices
4. [query_faiss.py](query_faiss.py) - Demo query with answer generation

### Option 2: Step-by-Step (Recommended for Development)

```bash
# Step 1: Download and prepare data
python data_preparation.py

# Step 2: Exploratory analysis (optional)
python eda_medvidqa.py

# Step 3: Process videos and build indices
python multimodal_pipeline_with_sliding_window.py

# Step 4: Query the system
python query_faiss.py --query "How to check for mouth cancer at home?" --split test --generate_answer --enable_curation
```

### Option 3: Quick Demo (Skip Video Processing)

If you already have FAISS indices:

```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split train \
  --hybrid \
  --alpha 0.3 \
  --enable_curation \
  --enable_attribution \
  --generate_answer
```

## 📖 Detailed Usage Guide

### 1. Data Preparation

**Script**: [data_preparation.py](data_preparation.py)

**What it does**:

- Downloads videos from YouTube using metadata in [MedVidQA](MedVidQA)
- Cleans dataset (removes duplicates, validates timestamps)
- Filters by successfully extracted features
- Organizes into [videos_train](videos_train), [videos_val](videos_val), [videos_test](videos_test)

**Usage**:

```bash
# Standard run
python data_preparation.py

# Custom options (edit script variables)
MODEL_NAME = "openai/whisper-tiny"  # ASR model for filtering
```

**Output**:

- [videos_train](videos_train) - Training videos (800 videos)
- [videos_val](videos_val) - Validation videos (49 videos)
- [videos_test](videos_test) - Test videos (50 videos)
- [MedVidQA_cleaned](MedVidQA_cleaned) - Cleaned JSON metadata

---

### 2. Exploratory Data Analysis

**Script**: [eda_medvidqa.py](eda_medvidqa.py)

**What it does**:

- Analyzes dataset statistics (video length distribution, question types)
- Visualizes temporal patterns (answer duration, coverage)
- Generates plots saved to `EDA/`

**Usage**:

```bash
python eda_medvidqa.py
```

**Output**:

- Console statistics (video counts, average durations)
- Plots in `EDA/`:
  - Video length distribution
  - Answer duration distribution
  - Question length histogram
  - Temporal coverage analysis

---

### 3. Multimodal Feature Extraction

**Script**: [multimodal_pipeline_with_sliding_window.py](multimodal_pipeline_with_sliding_window.py)

**What it does**:

- **Phase 2**: ASR transcription → Text/visual embedding extraction
- **Phase 3**: FAISS index construction

**Architecture**:

- Uses [VideoProcessor](video_processing/pipeline.py) for orchestration
- Modular components in [video_processing](video_processing):
  - [video_processing/asr.py](video_processing/asr.py) - Whisper ASR
  - [video_processing/text_embeddings.py](video_processing/text_embeddings.py) - Sliding window + BiomedCLIP text
  - [video_processing/visual_embeddings.py](video_processing/visual_embeddings.py) - Adaptive frame sampling + BiomedCLIP vision

**Usage**:

**Default Configuration** (recommended):

```bash
python multimodal_pipeline_with_sliding_window.py
```

**Custom Hyperparameters**:

```python
# Edit main() function in the script
process_split(
    split="train",
    video_dir="videos_train",
    text_feat_dir="feature_extraction/textual/train",
    visual_feat_dir="feature_extraction/visual/train",
    faiss_text_path="faiss_db/textual_train.index",
    faiss_visual_path="faiss_db/visual_train.index",

    # Text embedding parameters
    window_size=256,              # Token window size
    stride=192,                   # Overlap (192 = 75% overlap)
    min_coverage_contribution=0.05,  # Min new coverage to keep window
    deduplication_mode='coverage',   # 'coverage', 'similarity', 'aggressive', 'none'

    # Visual embedding parameters
    frames_per_segment=2,         # Frames extracted per segment
    sampling_strategy='adaptive',  # 'uniform', 'adaptive', 'quality_based'
    quality_filter=False,         # Enable frame quality filtering
    aggregation_method='mean'     # 'mean' or 'max' for multi-frame pooling
)
```

**Parallel Processing**:

```python
# Edit at top of script
ENABLE_PARALLEL = True  # Enable/disable multiprocessing
batch_size = 4          # Videos per batch (adjust for RAM)
max_workers = 2         # CPU/GPU workers (adjust for hardware)
```

**Output**:

- `feature_extraction/textual/{split}/` - Text embeddings JSON
- `feature_extraction/visual/{split}/` - Visual embeddings JSON
- `faiss_db/textual_{split}.index` - Text FAISS index
- `faiss_db/visual_{split}.index` - Visual FAISS index
- `faiss_db/*.meta.json` - Metadata files

**Performance Tips**:

- **GPU acceleration**: Automatically uses CUDA/MPS if available
- **Memory management**: Adjust `batch_size` if running out of RAM
- **Skip processed videos**: Set `skip_if_exists=True` (default)

---

### 4. FAISS Index Management

**Script**: [embedding_storage.py](embedding_storage.py)

**What it does**:

- Builds FAISS indices from JSON feature files
- Normalizes embeddings (L2 norm)
- Persists metadata alongside indices

**Usage**:

**Build indices for a split**:

```bash
# Build both text and visual indices
python embedding_storage.py --split train

# Build only text index
python embedding_storage.py --split train --modality text

# Build only visual index
python embedding_storage.py --split train --modality visual
```

**Custom directories**:

```bash
python embedding_storage.py \
  --split train \
  --feature_dir my_features/ \
  --output_dir my_indices/
```

**When to use**:

- After manual feature extraction
- To rebuild indices without reprocessing videos
- When experimenting with index configurations

---

### 5. Querying the System

**Script**: [query_faiss.py](query_faiss.py)

**What it does**:

- Embeds query with BiomedCLIP
- Searches FAISS indices (text + visual)
- Applies hybrid search (dense + BM25)
- Aggregates multimodal results
- Generates answers with GPT-4o-mini
- Evaluates against ground truth (if provided)

**Basic Usage**:

**Simple query** (dense search only):

```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split train \
  --final_k 10
```

**Hybrid search** (recommended):

```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split train \
  --hybrid \
  --alpha 0.3 \
  --fusion linear \
  --final_k 10
```

**With answer generation**:

```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split test \
  --hybrid \
  --enable_curation \
  --enable_attribution \
  --generate_answer
```

**Full evaluation with ground truth**:

```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split test \
  --hybrid \
  --eval \
  --video_id "_6csIJAWj_s" \
  --answer_start 35 \
  --answer_end 96 \
  --generate_answer \
  --enable_curation
```

**Advanced Options**:

**Search Configuration**:

```bash
--local_k 50          # Results per index before merging (default: 50)
--final_k 10          # Final combined results (default: 10)
--text_weight 0.6     # Text importance in multimodal fusion (default: 0.6)
--visual_weight 0.4   # Visual importance (default: 0.4)
```

**Hybrid Search**:

```bash
--hybrid              # Enable hybrid search
--alpha 0.3           # Dense weight (0-1, default: 0.3)
                      # 0.3 = 70% BM25 + 30% dense (optimal for medical)
--fusion linear       # 'linear' (score-based) or 'rrf' (rank-based)
--expand_query        # Medical term expansion in BM25 (default: enabled)
--analyze_fusion      # Show BM25 vs dense contribution breakdown
```

**Context Curation**:

```bash
--enable_curation           # Enable adaptive context selection
--quality_threshold 0.1     # Min quality score (default: 0.1)
--token_budget 600          # Max context tokens (default: 600)
--nli_top_k 15             # Candidates for NLI scoring (default: 15)
```

**Answer Generation**:

```bash
--generate_answer              # Enable LLM answer generation
--answer_model gpt-4o-mini    # Model (default: gpt-4o-mini)
--answer_max_tokens 250        # Max output length (default: 250)
--answer_temperature 0.3       # Sampling temp (default: 0.3)
--enable_attribution           # Self-reflection attribution
```

**Aggregation Modes**:

```bash
--mode segment        # Segment-level multimodal linking (default)
--mode video          # Video-level aggregation
--top_videos 5        # Videos in video mode (default: 5)
--hierarchical        # Extended context with overlapping timestamps
```

**Output**:

- Console: Search results, evaluation metrics, generated answer
- JSON: `multimodal_search_results_*.json` (search results + answer)

---

### 6. Batch Evaluation

**Script**: [run_full_evaluation.py](run_full_evaluation.py)

**What it does**:

- Evaluates entire dataset splits
- Computes retrieval and generation metrics
- Generates comprehensive reports
- Tracks performance and costs

**Usage**:

**Evaluate test split**:

```bash
python run_full_evaluation.py \
  --split test \
  --enable_curation \
  --enable_attribution
```

**Evaluate all splits** (train + val + test):

```bash
python run_full_evaluation.py \
  --split all \
  --alpha 0.3 \
  --quality_threshold 0.1
```

**Quick test** (first 10 queries):

```bash
python run_full_evaluation.py \
  --split test \
  --max_queries 10
```

**Custom configuration**:

```bash
python run_full_evaluation.py \
  --split test \
  --alpha 0.3 \
  --local_k 50 \
  --final_k 10 \
  --quality_threshold 0.1 \
  --token_budget 600 \
  --enable_curation \
  --enable_attribution
```

**Output**:

- `evaluation_results/` directory:
  - `evaluation_results_{split}_{timestamp}.json` - Detailed per-query results
  - `evaluation_summary__{split}_{timestamp}.json` - Aggregated statistics
  - `evaluation_report_{split}_{timestamp}.md` - Human-readable report
  - `checkpoint_{split}_{timestamp}_{query_count}`.json - For certain checkpoints in multiples of 50

**Metrics Reported**:

**Retrieval**:

- Recall@K
- Precision@K
- Mean Average Precision (mAP)
- lized Discounted Cumulative Gain (nDCG@K)
- Temporal IoU, Precision, Recall, F1
- Video hit rate

**Generation**:

- Answer quality (length, coherence)
- Confidence scores
- Token usage and costs

**Performance**:

- Search latency
- Generation latency
- End-to-end timing

---

### 7. Hyperparameter Tuning

**Script**: [hyperparameter_tuning.py](hyperparameter_tuning.py)

**What it does**:

- Grid search over embedding parameters
- Evaluates on metrics (coverage, relevance, speed)
- Generates comparison plots

**Usage**:

```bash
python hyperparameter_tuning.py
```

**Customize grid** (edit script):

```python
HYPERPARAMETER_GRID = {
    # Text parameters
    'window_size': [128, 256, 512],
    'stride': [96, 192, 384],
    'deduplication_mode': ['coverage', 'similarity', 'none'],

    # Visual parameters
    'frames_per_segment': [1, 2, 4],
    'sampling_strategy': ['uniform', 'adaptive', 'quality_based'],
    'aggregation_method': ['mean', 'max']
}
```

**Output**:

- `hyperparameter_tuning_results.json` - Detailed results
- `hyperparameter_comparison.png` - Visualization

---

### 8. Specialized Tools

#### 8.1 Rebuild Visual Embeddings

**Script**: [rebuild_visual_embeddings.py](rebuild_visual_embeddings.py)

**When to use**: Update visual features without reprocessing text

```bash
python rebuild_visual_embeddings.py --split train
```

#### 8.2 Compare Search Methods

**Script**: [compare_search_methods.py](compare_search_methods.py)

**What it does**: A/B test dense vs hybrid search

```bash
python compare_search_methods.py \
  --query "How to do a mouth cancer check at home?" \
  --expected_video "_6csIJAWj_s" \
  --top_k 10
```

#### 8.3 Answer Evaluation

**Script**: [evaluation.py](evaluation.py)

**What it does**: Standalone answer quality assessment

```python
from evaluation import AnswerEvaluator

evaluator = AnswerEvaluator()
result = evaluator.evaluate_answer(
    generated_answer="...",
    ground_truth="...",
    query="..."
)
```

---

## 📁 Project Structure

```
.
├── data_preparation.py                 # Download videos, clean dataset
├── eda_medvidqa.py                    # Exploratory data analysis
├── multimodal_pipeline_with_sliding_window.py  # Feature extraction pipeline
├── embedding_storage.py                # FAISS index management
├── query_faiss.py                     # Query interface (main entry point)
├── run_full_evaluation.py             # Batch evaluation
├── hyperparameter_tuning.py           # Hyperparameter optimization
├── compare_search_methods.py          # Search method comparison
├── evaluation.py                      # Answer evaluation metrics
├── rebuild_visual_embeddings.py       # Visual embedding rebuild
├── main.py                            # Full pipeline orchestrator
│
├── video_processing/                  # Modular video processing
│   ├── __init__.py
│   ├── asr.py                        # Whisper ASR
│   ├── text_embeddings.py            # Sliding window + BiomedCLIP text
│   ├── visual_embeddings.py          # Frame sampling + BiomedCLIP vision
│   ├── deduplication.py              # Embedding deduplication
│   └── pipeline.py                   # VideoProcessor orchestrator
│
├── search/                            # Search components
│   ├── __init__.py
│   ├── dense_search.py               # FAISS + embedding models
│   ├── sparse_search.py              # BM25 + query expansion
│   ├── hybrid_fusion.py              # Hybrid search engine
│   ├── aggregation.py                # Multimodal result merging
│   ├── hierarchical_search.py        # Extended context retrieval
│   └── utils.py                      # Helper functions
│
├── generation/                        # Answer generation
│   ├── __init__.py
│   ├── answer_generator.py           # GPT-4o-mini interface
│   ├── context_curator.py            # Adaptive context selection
│   ├── attribution.py                # Self-reflection attribution
│   └── prompts.py                    # Prompt templates
│
├── MedVidQA/                         # Original dataset
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── README.md
│
├── MedVidQA_cleaned/                 # Cleaned dataset
│   ├── train_openai_whisper_tiny_cleaned.json
│   ├── val_openai_whisper_tiny_cleaned.json
│   └── test_openai_whisper_tiny_cleaned.json
│
├── videos_train/                     # Training videos (800 videos)
├── videos_val/                       # Validation videos (49 videos)
├── videos_test/                      # Test videos (50 videos)
│
├── feature_extraction/               # Extracted embeddings
│   ├── textual/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── visual/
│       ├── train/
│       ├── val/
│       └── test/
│
├── faiss_db/                         # FAISS indices
│   ├── textual_train.index
│   ├── textual_train.index.meta.json
│   ├── textual_val.index
│   ├── textual_val.index.meta.json
│   ├── textual_test.index
│   ├── textual_test.index.meta.json
│   ├── visual_train.index
│   ├── visual_train.index.meta.json
│   ├── visual_val.index
│   ├── visual_val.index.meta.json
│   ├── visual_test.index
│   └── visual_test.index.meta.json
│
├── evaluation_results/               # Evaluation outputs
│   ├── evaluation_results_*.json
│   ├── evaluation_summary_*.json
│   └── evaluation_report_*.md
|
├── EDA/                              # Contains visualizations
|
├── images/                           # Phase wise and system architecture
│   ├── phase_1.png
│   ├── phase_2.png
│   ├── phase_3.png
│   ├── phase_4.png
│   ├── phase_5.png
|   ├── phase_6.png
│   ├── System_Architecture.png
│
├── requirements.txt                  # Core dependencies
├── .env                              # API keys (create this)
├── .gitignore
```

---

## 🔧 Configuration Options

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=sk-...                 # OpenAI API key

# Optional
HF_TOKEN=hf_...                       # HuggingFace token (for BiomedCLIP)
```

### Common Hyperparameters

**Text Embeddings**:

- `window_size=256` - Token window size
- `stride=192` - Overlap (75% overlap = stride/window)
- `min_coverage_contribution=0.05` - Min new coverage threshold
- `deduplication_mode='coverage'` - 'coverage', 'similarity', 'aggressive', 'none'

**Visual Embeddings**:

- `frames_per_segment=2` - Frames extracted per segment
- `sampling_strategy='adaptive'` - 'uniform', 'adaptive', 'quality_based'
- `quality_filter=False` - Enable sharpness-based filtering
- `aggregation_method='mean'` - 'mean' or 'max' for frame pooling

**Search**:

- `alpha=0.3` - Dense weight in hybrid search (0.3 = 70% BM25 + 30% dense)
- `fusion='linear'` - 'linear' (score-based) or 'rrf' (rank-based)
- `local_k=50` - Results per index
- `final_k=10` - Final combined results

**Context Curation**:

- `quality_threshold=0.1` - Min relevance score
- `token_budget=600` - Max context tokens
- `nli_top_k=15` - Candidates for NLI scoring

**Answer Generation**:

- `answer_model='gpt-4o-mini'` - LLM model
- `answer_max_tokens=250` - Max answer length (~200 words)
- `answer_temperature=0.3` - Lower = more deterministic

---

## 🎯 Example Workflows

### Workflow 1: First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Download and prepare data
python data_preparation.py

# 4. Process videos (this takes time!)
python multimodal_pipeline_with_sliding_window.py

# 5. Test with a query
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split train \
  --hybrid \
  --generate_answer
```

### Workflow 2: Development/Experimentation

```bash
# Test on a single video
python -c "
from multimodal_pipeline_with_sliding_window import test_single_video
test_single_video(
    'videos_test/sample.mp4',
    'feature_extraction/textual/test_single',
    'feature_extraction/visual/test_single'
)
"

# Query with different fusion strategies
python query_faiss.py --query "..." --split test --hybrid --alpha 0.1  # More BM25
python query_faiss.py --query "..." --split test --hybrid --alpha 0.5  # Balanced
python query_faiss.py --query "..." --split test --hybrid --alpha 0.9  # More dense

# Compare search methods
python compare_search_methods.py --query "..." --expected_video "..."
```

### Workflow 3: Batch Evaluation

```bash
# Evaluate with different configurations
python run_full_evaluation.py --split test --alpha 0.1
python run_full_evaluation.py --split test --alpha 0.3
python run_full_evaluation.py --split test --alpha 0.5

# Compare results
diff evaluation_results/test_summary_*_alpha0.1.json \
     evaluation_results/test_summary_*_alpha0.3.json
```

### Workflow 4: Hyperparameter Optimization

```bash
# 1. Define grid in hyperparameter_tuning.py
# 2. Run grid search
python hyperparameter_tuning.py

# 3. Analyze results
# Check hyperparameter_tuning_results.json
# View hyperparameter_comparison.png

# 4. Update pipeline with best hyperparameters
# Edit multimodal_pipeline_with_sliding_window.py
```

---

## 📊 Performance Benchmarks

### Hardware Performance (899 videos)

| Hardware               | Feature Extraction Time | Index Build Time | Query Latency |
| ---------------------- | ----------------------- | ---------------- | ------------- |
| CPU Only (16 cores)    | ~36-48 hours            | ~5 min           | ~1-2s         |
| NVIDIA RTX 3080 (10GB) | ~8-10 hours             | ~5 min           | ~0.5-1s       |
| Apple M1 Max (64GB)    | ~10-12 hours            | ~5 min           | ~0.5-1s       |

### Retrieval Performance (Test Set, 102 queries)

### Ranking Metrics

| Metric      | Mean   | Std    | Median | Min    | Max    |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| Recall@5    | 0.6471 | 0.4779 | 1.0000 | 0.0000 | 1.0000 |
| Recall@10   | 0.6471 | 0.4779 | 1.0000 | 0.0000 | 1.0000 |
| Precision@5 | 0.1569 | 0.1361 | 0.2000 | 0.0000 | 0.6000 |
| mAP         | 0.3961 | 0.4032 | 0.2500 | 0.0000 | 1.0000 |
| nDCG@10     | 0.4604 | 0.4019 | 0.4307 | 0.0000 | 1.0000 |

### Temporal Accuracy

| Metric             | Mean   | Std    | Median |
| ------------------ | ------ | ------ | ------ |
| IoU                | 0.0751 | 0.0703 | 0.0564 |
| Temporal Precision | 0.1834 | 0.1802 | 0.1261 |
| Temporal Recall    | 0.1254 | 0.1230 | 0.0940 |
| Temporal F1        | 0.1321 | 0.1164 | 0.1068 |

**Correct Video Retrieved Rate:** 96.08%

---

Context Curation Performance

| Metric                 | Mean   | Std    | Median  |
| ---------------------- | ------ | ------ | ------- |
| Pass Rate              | 95.98% | 0.1323 | 100.00% |
| Reduction Rate         | 85.49% | 0.0517 | 90.00%  |
| Avg Conflicts Detected | 3.25   | 3.43   | 2       |

---

Answer Generation Performance

| Metric                | Mean      | Std       | Median    |
| --------------------- | --------- | --------- | --------- |
| Confidence            | 32.18%    | 0.1636    | 30.00%    |
| Answer Length (words) | 122.8     | 67.6      | 161       |
| Tokens Used           | 562.6     | 113.9     | 586       |
| Cost per Query        | $0.000160 | $0.000059 | $0.000193 |

**Answer Generation Success Rate:** 100.00%
**Total Estimated Cost:** $0.0163

---

Performance Timing

| Metric              | Mean | Std  | Median |
| ------------------- | ---- | ---- | ------ |
| Search Time (s)     | 0.18 | 0.16 | 0.15   |
| Generation Time (s) | 8.80 | 3.34 | 9.98   |
| Total Time (s)      | 8.98 | 3.34 | 10.14  |

**Total Evaluation Time:** 15.3 minutes

---

## Summary

- **Queries Processed:** 102
- **Average Performance:** 13.21% Temporal F1
- **Answer Quality:** 32.18% Confidence
- **Cost Efficiency:** $0.000160 per query

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```python
# Reduce batch size in multimodal_pipeline_with_sliding_window.py
batch_size = 2  # Default: 4
```

**2. FAISS Index Not Found**

```bash
# Rebuild indices
python embedding_storage.py --split train --modality both
```

**3. OpenAI API Rate Limit**

```python
# Add delay in run_full_evaluation.py
import time
time.sleep(1)  # Between queries
```

**4. YouTube Download Fails**

```bash
# Update youtube-po-token-generator
npm update -g youtube-po-token-generator

# Or use pytubefix with authentication
# See: https://github.com/JuanBindez/pytubefix
```

**5. SciSpacy Model Not Found**

```bash
# Reinstall SciSpacy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

## 📄 License

This project is for research and educational purposes. Video content is subject to original YouTube creators' licenses.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Support for additional embedding models (e.g., PubMedBERT)
- Alternative LLMs (Claude, Gemini, LLaMA)
- Advanced fusion strategies
- Multi-hop reasoning
- Interactive UI/API

**Built with ❤️ for the medical AI research community**
