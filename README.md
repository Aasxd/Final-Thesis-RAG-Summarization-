# Final-Thesis-RAG-Summarization-
RAG-based financial news summarisation system — MSc Data Science, LJMU 2026

# Financial News Summarisation Using Retrieval-Augmented Generation

MSc Data Science Dissertation — Liverpool John Moores University — March 2026  
Author: Mohammed Asad Khan

## Overview

This repository contains the full implementation of a Retrieval-Augmented Generation 
(RAG) pipeline designed for financial text question answering and summarisation. The 
system was built and evaluated as part of an MSc dissertation investigating whether 
retrieval augmentation can meaningfully reduce hallucination in financial NLP tasks 
using only open-weight models and consumer-grade hardware.

The system retrieves relevant passages from a 64,294-chunk financial corpus at query 
time and conditions response generation on that evidence, rather than relying solely 
on the language model's parametric memory. A controlled comparison against a baseline 
LLM (same model, no retrieval) demonstrated an 82% reduction in unsupported causal 
inference and complete elimination of numerical fabrication.

## System Architecture

- Embedding model: all-MiniLM-L6-v2 (384-dimensional dense vectors)
- Vector index: FAISS IndexFlatIP (exact cosine search)
- Generator: Microsoft Phi-3-mini-4k-instruct
- Hardware: NVIDIA RTX 3060 Laptop GPU (6GB VRAM), AMD Ryzen 5000, 16GB RAM

## Datasets

All datasets used are publicly available:

- FinQA (Zhu et al., 2021): https://github.com/czyssrs/FinQA
- FinancialPhraseBank (Malo et al., 2014): https://huggingface.co/datasets/lmassaron/FinancialPhraseBank
- Financial News Headlines (CNBC, Reuters, The Guardian): https://www.kaggle.com/datasets/notlucasp/financial-news-headlines

## Repository Structure

Notebook_1_Data_Prep.ipynb  — data loading, chunking, embedding, FAISS index
Notebook_2_RAG.ipynb        — RAG pipeline, baseline comparison, evaluation

## Requirements

pip install -r requirements.txt

Key dependencies: pandas, numpy, torch, transformers, sentence-transformers, faiss-cpu, accelerate, jupyter
 

## How to Run

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run notebook_1_Data_Prep to build the corpus and FAISS index
4. Run notebook_2_RAG to run the RAG evaluation and baseline comparison

## Key Results

- 82% reduction in unsupported causal inference (22 → 4 instances)
- Numerical fabrication eliminated entirely (1 baseline instance → 0 RAG)
- 65% reduction in contextual overgeneralisation (17 → 6 of 20 queries)
- Conservative fallback behaviour observed 3 times in RAG, 0 times in baseline

## Citation

Khan, M.A. (2025) Summarization of Financial News Using Retrieval-Augmented 
Generation. MSc Dissertation. Liverpool John Moores University.
