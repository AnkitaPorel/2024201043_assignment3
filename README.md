# Word Embedding Models and Evaluation

This repository contains Python scripts to train word embeddings using different methods (SVD-based PPMI, Skip-gram, and CBOW) on the Brown Corpus and evaluate their quality using the WordSim-353 dataset. The scripts generate word embeddings, save them as PyTorch files, and assess their performance by computing cosine similarity against human similarity judgments.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Scripts](#scripts)
  - [SVD-based PPMI Embeddings](#svd-based-ppmi-embeddings)
  - [Evaluation Script](#evaluation-script)
  - [Skip-gram Model](#skip-gram-model)
  - [CBOW Model](#cbow-model)
- [Usage](#usage)
- [Output](#output)
- [Notes](#notes)
- [License](#license)

## Overview
Word embeddings are dense vector representations of words that capture semantic relationships. This project implements three methods:
1. **SVD-based PPMI**: Uses a co-occurrence matrix with Positive Pointwise Mutual Information (PPMI) and Singular Value Decomposition (SVD).
2. **Skip-gram**: A neural network model with negative sampling to predict context words from target words.
3. **CBOW**: A neural network model to predict a target word from its context words.

The evaluation script (`wordsim.py`) compares these embeddings against human similarity scores from WordSim-353, computing Spearman correlation and visualizing the results.

## Requirements
- Python 3.1+
- Libraries:
  - `numpy`
  - `torch`
  - `scipy`
  - `nltk`
  - `matplotlib`
  - `pandas`

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>