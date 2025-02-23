# Word Embedding Model Comparison: SVD, CBOW, and Skip-gram

## Introduction
Word embeddings are vector representations of words that capture semantic and syntactic relationships, widely used in natural language processing (NLP). This report compares three embedding methods—Singular Value Decomposition (SVD), Continuous Bag of Words (CBOW), and Skip-gram—trained on the Brown Corpus and evaluated against human similarity judgments from the WordSim-353 dataset. The evaluation metric is Spearman rank correlation, which measures how well the cosine similarity between word vectors aligns with human-assigned similarity scores.

The results are as follows:
- **SVD**: Spearman correlation = 0.127
- **CBOW**: Spearman correlation = 0.202
- **Skip-gram**: Spearman correlation = 0.387

This report details the methodology, implementation specifics from the provided scripts, and an analysis of these results.

---

## Methodology

### Dataset
- **Training Corpus**: The Brown Corpus (~1 million words) from NLTK, containing diverse English texts.
- **Evaluation Dataset**: WordSim-353 (`wordsim353crowd.csv`), a set of 353 word pairs with human-assigned similarity scores (0–10).

### Models

#### 1. SVD (`svd.py`)
- **Approach**: Constructs a co-occurrence matrix and uses SVD to reduce dimensionality into word embeddings.
- **Preprocessing**: Lowercases text, removes punctuation, and maps rare words (frequency < 2) to an `UNK` token.
- **Co-occurrence Matrix**: Built with a window size of 5, counting raw occurrences of word-context pairs within this range.
- **Dimensionality Reduction**: SVD reduces the matrix to 150 dimensions, producing embeddings without additional normalization.
- **Output**: Embeddings saved as `svd.pt` in a `{word: tensor}` dictionary.

#### 2. CBOW (`cbow.py`)
- **Approach**: A neural network model predicting a target word from its context words, trained with cross-entropy loss.
- **Preprocessing**: Lowercases text and removes punctuation (stopwords not removed in this implementation).
- **Training Data**: Context-target pairs with a window size of 5, padded with `UNK` if context is smaller than expected.
- **Model**: Embedding layer (300 dimensions) followed by a linear layer, averaging context embeddings to predict the target word.
- **Training**: 50 epochs, batch size 512, learning rate 0.0005, Adam optimizer, on CPU or GPU if available.
- **Output**: Embeddings saved as `cbow.pt`.

#### 3. Skip-gram (`skipgram.py`)
- **Approach**: A neural network model predicting context words from a target word, using negative sampling to optimize efficiency.
- **Preprocessing**: Lowercases text, removes punctuation (stopwords removal commented out), and applies subsampling to frequent words.
- **Training Data**: Target-context pairs with a window size of 5, subsampled based on word frequency.
- **Model**: Two embedding layers (target and context, 100 dimensions), trained with negative sampling (5 negative samples per positive).
- **Training**: 15 epochs, batch size 512, learning rate 0.001, Adam optimizer, on CPU or GPU if available.
- **Output**: Embeddings saved as `skipgram.pt`.

### Evaluation (`wordsim.py`)
- **Method**: Loads embeddings, computes cosine similarity for WordSim-353 word pairs, and calculates Spearman correlation with human scores.
- **Normalization**: Human scores normalized to [0, 1] (divided by 10) for plotting.
- **Output**: Spearman correlation, CSV file (`word_similarity.csv`), and scatter plot (`word_similarity_plot.png`).

---

## Implementation Details

### SVD
- **Vocabulary**: Minimum frequency 2, resulting in a vocabulary size of ~15,000–20,000 words (exact size printed during execution).
- **Matrix**: Sparse `lil_matrix` for efficiency, capturing raw co-occurrence counts.
- **SVD**: Uses `scipy.sparse.linalg.svds` with `k=150`, producing 150-dimensional embeddings.

### CBOW
- **Vocabulary**: Minimum frequency 5, resulting in a smaller vocabulary (~10,000–15,000 words).
- **Training Data**: Fixed context size (10 words, padded with `UNK`), leading to ~500,000–600,000 training samples.
- **Model**: Simple architecture with higher dimensionality (300) to capture more semantic nuance.

### Skip-gram
- **Vocabulary**: Minimum frequency 5, similar to CBOW.
- **Training Data**: Variable context size (up to 10 words), subsampling reduces sample count (~300,000–400,000 pairs).
- **Negative Sampling**: Noise distribution based on unigram frequency raised to 0.75, enhancing training efficiency.

---

## Results

### Spearman Correlation Scores
- **SVD**: 0.127
- **CBOW**: 0.202
- **Skip-gram**: 0.387

### Analysis

#### 1. SVD (0.127)
- **Performance**: The lowest correlation, indicating poor alignment with human judgments.
- **Reasons**:
  - **Small Corpus**: The Brown Corpus (~1M words) provides limited co-occurrence data, leading to a sparse matrix with unreliable counts.
  - **Raw Counts**: Using unweighted co-occurrence counts may overemphasize frequent but uninformative word pairs.
  - **Global Approach**: SVD captures global patterns across the entire corpus, potentially missing local context nuances critical for similarity tasks.
- **Observation**: The low score suggests that basic SVD struggles to generate meaningful embeddings from a small dataset without additional weighting or refinement.

#### 2. CBOW (0.202)
- **Performance**: Moderate improvement over SVD, but still relatively low.
- **Reasons**:
  - **Context Averaging**: CBOW averages context embeddings, which may dilute specific word relationships, especially with a small corpus and large window (5).
  - **Higher Dimensionality**: 300 dimensions might overfit or fail to generalize well with limited data.
  - **Training**: 50 epochs may not suffice for optimal convergence on a small dataset.
- **Observation**: CBOW captures some semantic structure better than SVD, likely due to its neural optimization, but its performance remains limited.

#### 3. Skip-gram (0.387)
- **Performance**: The highest correlation, significantly outperforming SVD and CBOW.
- **Reasons**:
  - **Local Context**: Skip-gram predicts context from target words, excelling at capturing local syntactic and semantic relationships.
  - **Subsampling**: Reduces noise from frequent words, improving embedding quality.
  - **Negative Sampling**: Efficiently contrasts positive and negative contexts, enhancing discriminative power.
  - **Fewer Epochs**: 15 epochs with a lower learning rate (0.001) suggest effective optimization without overfitting.
- **Observation**: Skip-gram’s superior performance aligns with its known strength in smaller corpora and its focus on individual word-context relationships.

### Comparative Analysis
- **Ranking**: Skip-gram (0.387) > CBOW (0.202) > SVD (0.127).
- **Trends**:
  - Neural models (Skip-gram, CBOW) outperform the statistical SVD approach, likely due to iterative optimization capturing nuanced patterns.
  - Skip-gram’s focus on predicting context outperforms CBOW’s context-to-target prediction, consistent with prior research on smaller datasets.
- **Corpus Size Impact**: The Brown Corpus’s limited size (~1M words) hampers all models, but Skip-gram adapts best, followed by CBOW, with SVD suffering most from sparse data.
- **Dimensionality**: Skip-gram (100) and SVD (150) use lower-to-moderate dimensions compared to CBOW (300), suggesting that higher dimensionality doesn’t guarantee better performance with limited data.

---

## Discussion
- **Skip-gram Success**: Its 0.387 correlation, while modest compared to modern embeddings (e.g., Word2Vec on larger corpora often exceeds 0.6), is respectable for the Brown Corpus. Subsampling and negative sampling optimize it for small datasets.
- **CBOW Limitation**: The 0.202 score reflects CBOW’s reliance on broader context averaging, which may lose specificity in a small corpus.
- **SVD Weakness**: The 0.127 score indicates that basic SVD with raw co-occurrence counts struggles with sparse, small-scale data, where frequency patterns are less reliable.
- **Improvement Opportunities**:
  - **Larger Corpus**: Training on a bigger dataset (e.g., Wikipedia) could boost all scores.
  - **Hyperparameter Tuning**: Adjusting window size, embedding dimension, or epochs might improve results.
  - **Preprocessing**: Adding stopword removal or lemmatization could refine embeddings, especially for SVD.

---

## Conclusion
Skip-gram outperforms CBOW and SVD in this implementation, achieving a Spearman correlation of 0.387, followed by CBOW (0.202) and SVD (0.127). These results highlight Skip-gram’s robustness for smaller corpora, CBOW’s moderate capability, and SVD’s limitations with unweighted co-occurrence data. For practical NLP tasks with the Brown Corpus, Skip-gram embeddings are recommended, though all models could benefit from a larger training corpus and further optimization.

---