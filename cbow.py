import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import brown
import nltk
import string
from collections import Counter
from scipy.stats import spearmanr

nltk.download('brown')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess(sentence):
    """Preprocess a sentence: lowercase, remove punctuation, and optionally remove stopwords."""
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in sentence.split()]
    return " ".join(words)

def build_vocab(corpus, min_freq=5):
    word_counts = Counter()
    for sentence in corpus:
        words = sentence.split()
        word_counts.update(words)

    vocab = {'UNK': 0}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def generate_training_data(corpus, vocab, window_size=2):
    context_size = 2 * window_size
    training_data = []
    for sentence in corpus:
        words = sentence.split()
        for i, target_word in enumerate(words):
            target_id = vocab.get(target_word, vocab['UNK'])
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words = [words[j] for j in range(start, end) if j != i]
            context_ids = [vocab.get(word, vocab['UNK']) for word in context_words]
            while len(context_ids) < context_size:
                context_ids.append(vocab['UNK'])
            training_data.append(context_ids + [target_id])

    training_data = np.array(training_data, dtype=np.int64)
    return torch.tensor(training_data).to(device)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        context_ids = inputs[:, :-1]
        embedded = self.embeddings(context_ids)
        context_mean = torch.mean(embedded, dim=1)
        out = self.linear(context_mean)
        return out

def train_cbow(model, training_data, epochs=50, batch_size=512, learning_rate=0.0005):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    num_samples = training_data.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        indices = torch.randperm(num_samples).to(device)
        total_loss = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch = training_data[batch_indices]
            context_ids = batch[:, :-1]
            target_ids = batch[:, -1]

            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_num = i // batch_size + 1
            if batch_num % max(1, num_batches // 10) == 0 or i + batch_size >= num_samples:
                progress = (batch_num / num_batches) * 100
                print(f"  Progress: {progress:.1f}% ({batch_num}/{num_batches} batches), "
                      f"Batch Loss: {loss.item():.4f}, Avg Loss so far: {total_loss / batch_num:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    return model

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_word_similarity(embeddings, wordsim_path):
    human_scores = []
    model_scores = []
    missing = 0
    with open(wordsim_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        word1, word2, score = line.strip().split(',')
        if word1 in embeddings and word2 in embeddings:
            sim = cosine_similarity(embeddings[word1], embeddings[word2])
            human_scores.append(float(score))
            model_scores.append(sim)
        else:
            missing += 1
    print(f"Evaluated {len(human_scores)} pairs, {missing} missing from vocab")
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation

def main():
    sentences = brown.sents()
    processed_sentences = [preprocess(" ".join(sentence)) for sentence in sentences]

    vocab = build_vocab(processed_sentences, min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")
    unk_count = sum(1 for s in processed_sentences for w in s.split() if w not in vocab)
    total_words = sum(len(s.split()) for s in processed_sentences)
    print(f"UNK frequency: {unk_count} ({unk_count / total_words * 100:.2f}%)")

    training_data = generate_training_data(processed_sentences, vocab, window_size=5)
    print(f"Training data size: {training_data.size(0)}")

    embedding_dim = 300
    model = CBOW(vocab_size=len(vocab), embedding_dim=embedding_dim)
    trained_model = train_cbow(model, training_data, epochs=50, batch_size=512, learning_rate=0.0005)

    embeddings = trained_model.embeddings.weight.data.cpu().numpy()
    embeddings_dict = {word: embeddings[idx] for word, idx in vocab.items()}
    torch.save(embeddings_dict, '/content/cbow.pt')
    print("CBOW embeddings saved to 'cbow.pt'")

    cbow_embeddings = torch.load('/content/cbow.pt')
    wordsim_path = '/content/wordsim353crowd.csv'
    cbow_correlation = evaluate_word_similarity(cbow_embeddings, wordsim_path)
    print(f"CBOW Spearman Correlation: {cbow_correlation:.4f}")

if __name__ == "__main__":
    main()