#!/bin/bash
# Quick test script for hybrid search

echo "=========================================="
echo "HYBRID SEARCH - QUICK TEST"
echo "=========================================="

# Test query
QUERY="How to do a mouth cancer check at home?"
EXPECTED_VIDEO="_6csIJAWj_s"

echo ""
echo "Query: $QUERY"
echo "Expected Video: $EXPECTED_VIDEO"
echo ""

# Run comparison
python compare_search_methods.py \
  --query "$QUERY" \
  --expected_video "$EXPECTED_VIDEO" \
  --top_k 10 \
  --alpha 0.7

echo ""
echo "=========================================="
echo "EXPERIMENT WITH DIFFERENT SETTINGS"
echo "=========================================="
echo ""
echo "Try different alpha values:"
echo "  --alpha 0.5  # 50% dense, 50% BM25 (more keyword matching)"
echo "  --alpha 0.8  # 80% dense, 20% BM25 (more semantic)"
echo ""
echo "Try different fusion methods:"
echo "  --fusion linear  # Score-based fusion (default)"
echo "  --fusion rrf     # Reciprocal Rank Fusion"
echo ""
echo "View detailed analysis:"
echo "  python query_faiss.py --query \"$QUERY\" --hybrid --analyze_fusion"
