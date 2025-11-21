"""
hybrid_search.py

Hybrid search combining BM25 (sparse lexical) with dense embeddings for improved retrieval.

Key features:
- BM25 for keyword matching (captures exact medical terminology)
- Dense embeddings for semantic similarity
- Configurable fusion strategies (linear, RRF)
- Medical query expansion support

Usage:
    from hybrid_search import HybridSearchEngine

    engine = HybridSearchEngine(text_index_path, segments_data)
    results = engine.search(query, top_k=10, alpha=0.7)
"""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
import re


class HybridSearchEngine:
    """
    Hybrid search engine combining BM25 (sparse) and dense embeddings.

    Args:
        segments_data: List of segment dictionaries with 'text' and 'segment_id' fields
        alpha: Weight for dense retrieval (0-1), (1-alpha) for BM25
        bm25_k1: BM25 term frequency saturation parameter (default: 1.5)
        bm25_b: BM25 length normalization parameter (default: 0.75)
    """

    def __init__(self, segments_data: List[Dict], alpha: float = 0.7,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.segments_data = segments_data
        self.alpha = alpha  # Weight for dense retrieval
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # Build BM25 index
        print(f"Building BM25 index for {len(segments_data)} segments...")
        self.corpus = [seg.get('text', '') for seg in segments_data]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=bm25_k1, b=bm25_b)

        # Create segment_id to index mapping
        self.segment_id_to_idx = {
            seg.get('segment_id', seg.get('video_id', f"seg_{i}")): i
            for i, seg in enumerate(segments_data)
        }

        print(f"✅ BM25 index built with {len(self.corpus)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Preserves medical terms and handles hyphenation.
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation, but preserve hyphenated terms
        tokens = re.findall(r'\b[\w-]+\b', text)

        return tokens

    def expand_medical_query(self, query: str) -> str:
        """
        Expand query with common medical synonyms from medical ontologies.
        Based on UMLS, SNOMED CT, and common medical terminology.
        """
        # Comprehensive medical term expansion based on medical ontologies
        expansions = {
            # Cancer and Oncology
            "mouth cancer": ["oral cancer", "oral cavity cancer", "oropharyngeal cancer", "mouth tumor"],
            "cancer": ["tumor", "malignancy", "carcinoma", "neoplasm", "oncology", "malignant growth"],
            "breast cancer": ["mammary carcinoma", "breast tumor", "breast malignancy"],
            "lung cancer": ["pulmonary carcinoma", "bronchogenic carcinoma", "lung tumor"],
            "skin cancer": ["melanoma", "basal cell carcinoma", "squamous cell carcinoma", "dermatologic cancer"],
            "tumor": ["neoplasm", "growth", "mass", "lesion", "malignancy"],

            # Cardiovascular
            "heart attack": ["myocardial infarction", "MI", "cardiac infarction", "coronary thrombosis"],
            "heart disease": ["cardiovascular disease", "cardiac disease", "coronary artery disease", "CAD"],
            "stroke": ["cerebrovascular accident", "CVA", "brain attack", "cerebral infarction"],
            "blood pressure": ["hypertension", "BP", "arterial pressure", "blood pressure level"],
            "high blood pressure": ["hypertension", "HTN", "elevated blood pressure", "arterial hypertension"],
            "heart failure": ["cardiac failure", "congestive heart failure", "CHF", "cardiac insufficiency"],
            "arrhythmia": ["irregular heartbeat", "cardiac dysrhythmia", "abnormal heart rhythm"],

            # Diabetes and Endocrine
            "diabetes": ["diabetic", "blood sugar", "glucose", "diabetes mellitus", "hyperglycemia"],
            "type 1 diabetes": ["insulin dependent diabetes", "IDDM", "juvenile diabetes"],
            "type 2 diabetes": ["non-insulin dependent diabetes", "NIDDM", "adult onset diabetes"],
            "thyroid": ["thyroid gland", "thyroid disorder", "thyroid disease"],

            # Respiratory
            "asthma": ["bronchial asthma", "reactive airway disease", "asthmatic condition"],
            "COPD": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
            "pneumonia": ["lung infection", "pulmonary infection", "pneumonitis"],
            "breathing": ["respiration", "respiratory", "ventilation", "breath"],

            # Infectious Diseases
            "infection": ["inflammatory", "sepsis", "contamination", "infectious disease", "pathogen"],
            "bacterial infection": ["bacterial disease", "bacteremia", "bacterial contamination"],
            "viral infection": ["virus", "viral disease", "viremia"],
            "flu": ["influenza", "viral flu", "influenza virus"],
            "COVID": ["coronavirus", "SARS-CoV-2", "COVID-19", "coronavirus disease"],

            # Neurological
            "alzheimer": ["dementia", "cognitive decline", "alzheimer disease", "memory loss"],
            "parkinson": ["parkinson disease", "PD", "parkinsonian syndrome"],
            "seizure": ["convulsion", "epileptic seizure", "fit", "epilepsy"],
            "migraine": ["headache", "severe headache", "migrainous headache"],
            "headache": ["cephalgia", "head pain", "cranial pain"],

            # Gastrointestinal
            "stomach": ["gastric", "abdominal", "belly", "gastrointestinal"],
            "diarrhea": ["loose stools", "loose bowel movements", "gastroenteritis"],
            "constipation": ["irregular bowels", "difficulty passing stool", "obstipation"],
            "ulcer": ["peptic ulcer", "gastric ulcer", "duodenal ulcer", "stomach ulcer"],

            # Musculoskeletal
            "arthritis": ["joint inflammation", "arthritic", "osteoarthritis", "rheumatoid arthritis"],
            "fracture": ["broken bone", "bone break", "bone fracture"],
            "osteoporosis": ["bone loss", "decreased bone density", "brittle bones"],
            "back pain": ["lumbar pain", "spinal pain", "dorsalgia", "backache"],

            # General Symptoms
            "symptom": ["sign", "indication", "manifestation", "clinical feature"],
            "pain": ["discomfort", "ache", "soreness", "hurt", "painful", "hurting"],
            "fever": ["pyrexia", "high temperature", "febrile", "elevated temperature"],
            "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy"],
            "nausea": ["sick to stomach", "queasiness", "feeling sick", "nauseated"],
            "dizziness": ["vertigo", "lightheaded", "giddiness", "dizzy"],
            "swelling": ["edema", "inflammation", "swollen", "puffiness"],

            # Medical Procedures
            "surgery": ["procedure", "operation", "surgical intervention", "operative procedure"],
            "check": ["examination", "screening", "inspection", "assessment", "evaluation"],
            "home": ["self-examination", "self-check", "self-screening", "at-home"],
            "diagnosis": ["diagnostic", "identification", "detection", "diagnose"],
            "treatment": ["therapy", "intervention", "management", "care", "therapeutic"],
            "prescription": ["medication", "drug", "medicine", "pharmaceutical"],
            "vaccine": ["vaccination", "immunization", "shot", "inoculation"],
            "test": ["screening", "diagnostic test", "lab test", "examination", "assay"],
            "scan": ["imaging", "radiograph", "X-ray", "CT", "MRI"],
            "biopsy": ["tissue sample", "tissue examination", "histology"],

            # Mental Health
            "depression": ["depressive disorder", "major depression", "clinical depression"],
            "anxiety": ["anxiety disorder", "anxious", "nervousness", "worry"],
            "stress": ["psychological stress", "mental stress", "tension"],

            # Kidney and Urinary
            "kidney": ["renal", "kidney function", "nephrology"],
            "kidney disease": ["renal disease", "nephropathy", "kidney disorder"],
            "urinary": ["urine", "urination", "urologic", "bladder"],

            # Reproductive Health
            "pregnancy": ["pregnant", "gestation", "prenatal", "expecting"],
            "menstruation": ["period", "menstrual cycle", "menses", "monthly cycle"],

            # Skin Conditions
            "rash": ["skin rash", "dermatitis", "skin irritation", "eruption"],
            "eczema": ["atopic dermatitis", "dermatitis", "skin inflammation"],
            "psoriasis": ["psoriatic condition", "skin lesion", "plaque psoriasis"],

            # Blood Disorders
            "anemia": ["low blood count", "iron deficiency", "anemic", "low hemoglobin"],
            "bleeding": ["hemorrhage", "blood loss", "hemorrhaging"],

            # Vision and Hearing
            "vision": ["sight", "visual", "eyesight", "eye health"],
            "blind": ["blindness", "vision loss", "visual impairment"],
            "hearing": ["auditory", "audition", "hearing ability"],
            "deaf": ["deafness", "hearing loss", "hearing impairment"],

            # Nutrition and Lifestyle
            "diet": ["nutrition", "dietary", "eating habits", "nutritional"],
            "exercise": ["physical activity", "workout", "fitness", "activity"],
            "obesity": ["overweight", "excess weight", "obese", "weight problem"],
            "weight loss": ["losing weight", "weight reduction", "slimming"],

            # Emergency and Acute Care
            "emergency": ["urgent", "acute", "critical", "urgent care"],
            "trauma": ["injury", "traumatic injury", "wound", "physical trauma"],
            "poisoning": ["toxic", "toxicity", "intoxication", "poison"],
        }

        query_lower = query.lower()
        expanded_terms = [query]

        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)

        return " ".join(expanded_terms)

    def search_bm25(self, query: str, top_k: int = 100, expand_query: bool = True) -> List[Dict]:
        """
        Search using BM25 sparse retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            expand_query: Whether to expand query with medical synonyms

        Returns:
            List of results with segment_id and bm25_score
        """
        # Optionally expand query
        if expand_query:
            expanded_query = self.expand_medical_query(query)
            print(f"Original query: {query}")
            print(f"Expanded query: {expanded_query}")
            query_to_use = expanded_query
        else:
            query_to_use = query

        # Tokenize query
        tokenized_query = self._tokenize(query_to_use)

        # Get BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include documents with non-zero scores
                results.append({
                    'segment_id': self.segments_data[idx].get('segment_id',
                                                               self.segments_data[idx].get('video_id', f"seg_{idx}")),
                    'bm25_score': float(bm25_scores[idx]),
                    'segment_idx': int(idx),
                    'metadata': self.segments_data[idx]
                })

        return results

    def hybrid_search(self, query: str, dense_results: List[Dict],
                     top_k: int = 10, fusion: str = 'linear',
                     expand_query: bool = True, rrf_k: int = 60) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and dense retrieval.

        Args:
            query: Search query
            dense_results: Results from dense retrieval (FAISS) with 'raw_score' and 'meta' fields
            top_k: Number of final results to return
            fusion: Fusion strategy ('linear' or 'rrf' for Reciprocal Rank Fusion)
            expand_query: Whether to expand medical query terms
            rrf_k: Parameter for RRF (default: 60)

        Returns:
            List of hybrid results sorted by combined score
        """
        # Get BM25 results
        bm25_results = self.search_bm25(query, top_k=len(self.segments_data),
                                        expand_query=expand_query)

        print(f"\nHybrid Search Stats:")
        print(f"  BM25 results (non-zero scores): {len(bm25_results)}")
        print(f"  Dense results: {len(dense_results)}")
        print(f"  Fusion strategy: {fusion}")
        print(f"  Alpha (dense weight): {self.alpha}")

        if fusion == 'rrf':
            # Reciprocal Rank Fusion
            return self._fuse_rrf(dense_results, bm25_results, top_k, rrf_k)
        else:
            # Linear combination (score-based)
            return self._fuse_linear(dense_results, bm25_results, top_k)

    def _fuse_linear(self, dense_results: List[Dict], bm25_results: List[Dict],
                     top_k: int) -> List[Dict]:
        """
        Linear score-based fusion.

        Combined score = alpha * dense_score + (1-alpha) * bm25_score
        """
        # Normalize BM25 scores to [0, 1]
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            min_bm25 = min(r['bm25_score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25

            if bm25_range > 0:
                for r in bm25_results:
                    r['bm25_score_normalized'] = (r['bm25_score'] - min_bm25) / bm25_range
            else:
                for r in bm25_results:
                    r['bm25_score_normalized'] = 0.0

        # Normalize dense scores (convert distance to similarity)
        for r in dense_results:
            dist = r.get('raw_score', float('inf'))
            # Use exponential decay for better score distribution
            r['dense_score_normalized'] = np.exp(-dist) if np.isfinite(dist) else 0.0

        # Create mapping of segment_id to scores
        bm25_scores = {r['segment_id']: r['bm25_score_normalized'] for r in bm25_results}

        # Combine scores
        combined_results = {}

        # Process dense results
        for r in dense_results:
            meta = r.get('meta', {}) or {}
            segment_id = meta.get('segment_id', meta.get('video_id', 'unknown'))

            dense_score = r['dense_score_normalized']
            bm25_score = bm25_scores.get(segment_id, 0.0)

            combined_score = self.alpha * dense_score + (1 - self.alpha) * bm25_score

            combined_results[segment_id] = {
                'segment_id': segment_id,
                'combined_score': combined_score,
                'dense_score': dense_score,
                'bm25_score': bm25_score,
                'metadata': meta,
                'fusion_method': 'linear'
            }

        # Add BM25-only results (not in dense results)
        for r in bm25_results:
            segment_id = r['segment_id']
            if segment_id not in combined_results:
                bm25_score = r['bm25_score_normalized']
                combined_score = (1 - self.alpha) * bm25_score

                combined_results[segment_id] = {
                    'segment_id': segment_id,
                    'combined_score': combined_score,
                    'dense_score': 0.0,
                    'bm25_score': bm25_score,
                    'metadata': r['metadata'],
                    'fusion_method': 'linear'
                }

        # Sort by combined score
        sorted_results = sorted(combined_results.values(),
                               key=lambda x: x['combined_score'],
                               reverse=True)

        return sorted_results[:top_k]

    def _fuse_rrf(self, dense_results: List[Dict], bm25_results: List[Dict],
                  top_k: int, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) across all ranking methods

        Args:
            k: Constant for RRF (default: 60, standard in literature)
        """
        def get_rrf_score(rank: int, k: int) -> float:
            return 1.0 / (k + rank)

        combined_scores = {}

        # Process dense results (by rank)
        for rank, r in enumerate(dense_results):
            meta = r.get('meta', {}) or {}
            segment_id = meta.get('segment_id', meta.get('video_id', 'unknown'))

            if segment_id not in combined_scores:
                combined_scores[segment_id] = {
                    'segment_id': segment_id,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'bm25_rank': None,
                    'metadata': meta
                }

            combined_scores[segment_id]['rrf_score'] += self.alpha * get_rrf_score(rank, k)
            combined_scores[segment_id]['dense_rank'] = rank + 1

        # Process BM25 results (by rank)
        for rank, r in enumerate(bm25_results):
            segment_id = r['segment_id']

            if segment_id not in combined_scores:
                combined_scores[segment_id] = {
                    'segment_id': segment_id,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'bm25_rank': None,
                    'metadata': r['metadata']
                }

            combined_scores[segment_id]['rrf_score'] += (1 - self.alpha) * get_rrf_score(rank, k)
            combined_scores[segment_id]['bm25_rank'] = rank + 1

        # Sort by RRF score
        sorted_results = sorted(combined_scores.values(),
                               key=lambda x: x['rrf_score'],
                               reverse=True)

        # Add fusion method to results
        for r in sorted_results:
            r['combined_score'] = r['rrf_score']
            r['fusion_method'] = 'rrf'

        return sorted_results[:top_k]

    def analyze_fusion_contribution(self, hybrid_results: List[Dict], top_k: int = 10):
        """
        Analyze how BM25 and dense retrieval contribute to final results.
        Useful for tuning alpha parameter.
        """
        print(f"\n{'='*80}")
        print("FUSION ANALYSIS (Top {})".format(top_k))
        print(f"{'='*80}")

        for i, result in enumerate(hybrid_results[:top_k], 1):
            segment_id = result['segment_id']
            combined = result['combined_score']

            if result['fusion_method'] == 'linear':
                dense = result.get('dense_score', 0.0)
                bm25 = result.get('bm25_score', 0.0)

                dense_contrib = self.alpha * dense
                bm25_contrib = (1 - self.alpha) * bm25

                print(f"\n{i}. {segment_id}")
                print(f"   Combined: {combined:.4f}")
                print(f"   Dense contribution: {dense_contrib:.4f} (raw: {dense:.4f})")
                print(f"   BM25 contribution: {bm25_contrib:.4f} (raw: {bm25:.4f})")

                if dense > bm25:
                    print(f"   → Dense-driven result")
                elif bm25 > dense:
                    print(f"   → BM25-driven result")
                else:
                    print(f"   → Balanced result")

            else:  # RRF
                rrf = result.get('rrf_score', 0.0)
                dense_rank = result.get('dense_rank', 'N/A')
                bm25_rank = result.get('bm25_rank', 'N/A')

                print(f"\n{i}. {segment_id}")
                print(f"   RRF score: {rrf:.4f}")
                print(f"   Dense rank: {dense_rank}")
                print(f"   BM25 rank: {bm25_rank}")

        print(f"\n{'='*80}")


def load_segments_from_json_dir(json_dir: str = "feature_extraction/", split: str = None) -> List[Dict]:
    """
    Load all segments from JSON feature files, recursively searching through subdirectories.

    Args:
        json_dir: Base directory containing JSON files (default: 'feature_extraction/')
                  Will recursively search all subdirectories (train/test/val) for JSON files
        split: Optional split name to filter specific directory (e.g., 'train', 'test', 'val')
               If None, loads from all subdirectories

    Returns:
        List of segment dictionaries with text and metadata
    """
    import os
    import json

    segments = []

    # Handle specific split if provided
    if split:
        json_dir = os.path.join(json_dir, split)

    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    print(f"Loading segments from {json_dir} (searching recursively)...")

    # Recursively walk through all subdirectories
    json_files_found = 0
    for root, dirs, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                json_files_found += 1
                try:
                    with open(filepath, 'r') as f:
                        video_segments = json.load(f)

                        # Handle both list and dict formats
                        if isinstance(video_segments, list):
                            segments.extend(video_segments)
                        elif isinstance(video_segments, dict):
                            segments.append(video_segments)

                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")

    print(f"✅ Loaded {len(segments)} segments from {json_files_found} JSON files in {json_dir}")
    return segments
