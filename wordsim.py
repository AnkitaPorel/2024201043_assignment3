import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def evaluate_word_similarity(embeddings, wordsim_path):
#     human_scores = []
#     model_scores = []
#     word_pairs = []
#     with open(wordsim_path, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         word1, word2, score = line.strip().split(',')
#         if word1 in embeddings and word2 in embeddings:
#             sim = cosine_similarity(embeddings[word1], embeddings[word2])
#             human_scores.append(float(score))
#             model_scores.append(sim)
#             word_pairs.append((word1, word2))
#     correlation, _ = spearmanr(human_scores, model_scores)
#     return correlation, human_scores, model_scores, word_pairs

# def save_to_csv(word_pairs, human_scores, model_scores, output_path):
#     data = {
#         'Word1': [pair[0] for pair in word_pairs],
#         'Word2': [pair[1] for pair in word_pairs],
#         'Human_Score': human_scores,
#         'Model_Score': model_scores
#     }
#     df = pd.DataFrame(data)
#     df.to_csv(output_path, index=False)
#     print(f"Results saved to '{output_path}'")

# svd_embeddings = torch.load('/content/svd.pt')

# def plot_similarity(human_scores, model_scores, word_pairs):
#     human_scores_normalized = [score / 10.0 for score in human_scores]
#     plt.figure(figsize=(14, 12))
#     plt.scatter(human_scores_normalized, model_scores, alpha=0.5, color='blue', label='Word Pairs')

#     for i in range(len(word_pairs)):
#         plt.annotate(f"{word_pairs[i][0]}-{word_pairs[i][1]}", 
#                      (human_scores_normalized[i], model_scores[i]), 
#                      fontsize=5, alpha=0.6, xytext=(5, 5), textcoords='offset points')

#     plt.xlabel('Normalized Human Mean Similarity (0-1)')
#     plt.ylabel('Cosine Similarity (Model, -1 to 1)')
#     plt.title(f'Cosine Similarity vs Human Similarity (Spearman: {svd_correlation:.3f})')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()

#     plt.savefig('/content/word_similarity_plot.png', dpi=300, bbox_inches='tight')
#     print("Plot saved as '/content/word_similarity_plot.png'")
    
#     plt.show()

# wordsim_path = '/content/wordsim353crowd.csv'

# svd_correlation, human_scores, model_scores, word_pairs = evaluate_word_similarity(svd_embeddings, wordsim_path)
# print(f"SVD Spearman Correlation: {svd_correlation}")
# plot_similarity(human_scores, model_scores, word_pairs)

# csv_output_path = '/content/svd.csv'
# save_to_csv(word_pairs, human_scores, model_scores, csv_output_path)



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_word_similarity(embeddings, wordsim_path):
    human_scores = []
    model_scores = []
    word_pairs = []
    with open(wordsim_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        word1, word2, score = line.strip().split(',')
        if word1 in embeddings and word2 in embeddings:
            sim = cosine_similarity(embeddings[word1], embeddings[word2])
            human_scores.append(float(score))
            model_scores.append(sim)
            word_pairs.append((word1, word2))
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation, human_scores, model_scores, word_pairs

def plot_similarity(human_scores, model_scores, word_pairs):
    human_scores_normalized = [score / 10.0 for score in human_scores]
    plt.figure(figsize=(14, 12))
    plt.scatter(human_scores_normalized, model_scores, alpha=0.5, color='blue', label='Word Pairs')

    for i in range(len(word_pairs)):
        plt.annotate(f"{word_pairs[i][0]}-{word_pairs[i][1]}", 
                     (human_scores_normalized[i], model_scores[i]), 
                     fontsize=5, alpha=0.6, xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Normalized Human Mean Similarity (0-1)')
    plt.ylabel('Cosine Similarity (Model, -1 to 1)')
    plt.title(f'CBOW: Cosine Similarity vs Human Similarity (Spearman: {cbow_correlation:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig('/content/cbow.png', dpi=300, bbox_inches='tight')
    print("Plot saved as '/content/cbow.png'")
    
    plt.show()

def save_to_csv(word_pairs, human_scores, model_scores, output_path):
    data = {
        'Word1': [pair[0] for pair in word_pairs],
        'Word2': [pair[1] for pair in word_pairs],
        'Human_Score': human_scores,
        'Model_Score': model_scores
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")

cbow_embeddings = torch.load('/content/cbow.pt')
wordsim_path = '/content/wordsim353crowd.csv'
cbow_correlation, human_scores, model_scores, word_pairs = evaluate_word_similarity(cbow_embeddings, wordsim_path)
print(f"CBOW Spearman Correlation: {cbow_correlation}")
plot_similarity(human_scores, model_scores, word_pairs)

csv_output_path = '/content/cbow.csv'
save_to_csv(word_pairs, human_scores, model_scores, csv_output_path)

