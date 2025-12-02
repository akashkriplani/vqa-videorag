"""
main.py
High-level pipeline for Medical VideoRAG VQA.
"""

import data_preparation as dp
import eda_medvidqa as eda
import multimodal_pipeline_with_sliding_window as mmp
import query_faiss as qf

def data_preparation():
    """Load MedVidQA dataset and metadata, download videos, clean the dataset."""
    dp.main()

def data_analysis():
    """Perform exploratory data analysis (EDA) on cleaned datasets."""
    eda.main()

def run_multimodal_pipeline():
    """Run the multimodal VideoRAG VQA pipeline with sliding window approach."""
    mmp.main()

def query_faiss_and_generate_answer():
    """Query FAISS indices and generate answers."""
    qf.main()

def main():
    """Run the full pipeline."""
    data_preparation()
    data_analysis()
    run_multimodal_pipeline()
    query_faiss_and_generate_answer()

if __name__ == "__main__":
    main()
