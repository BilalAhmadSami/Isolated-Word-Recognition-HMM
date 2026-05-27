"""
hmm_testing.py
--------------
Evaluation of trained HMM word models using Viterbi decoding.

Loads pre-extracted MFCC descriptors and trained HMM models, runs Viterbi
decoding on each test utterance, and reports recognition results alongside
a confusion matrix over the 11-word vocabulary.

Module: EEEM030 – Speech & Audio Processing and Recognition
        University of Surrey
"""

import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- PATHS ----------------
DESCRIPTOR_PATH = "descriptors"
TEST_DESCRIPTOR = os.path.join(DESCRIPTOR_PATH, "test_descriptors.npy")
HMM_MODEL_PATH  = os.path.join(DESCRIPTOR_PATH, "hmm_models.pkl")

# ---------------- LOG-DOMAIN UTILITIES ----------------
EPS = 1e-6

def log_gaussian_prob(x, mean, var):
    """
    Log-probability of observation vector x under a diagonal Gaussian.
    """
    return -0.5 * np.sum(np.log(2 * np.pi * var + EPS) + (x - mean) ** 2 / (var + EPS))


def log_gaussian_prob_matrix(x, means, vars):
    """
    Log-probability of x under each of N diagonal Gaussians.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return np.array([log_gaussian_prob(x, means[i], vars[i]) for i in range(means.shape[0])])


def log_sum_exp(log_probs):
    """
    Numerically stable log-sum-exp over a vector of log-probabilities.
    """
    max_log = np.max(log_probs)
    return max_log + np.log(np.sum(np.exp(log_probs - max_log)))


# ---------------- VITERBI DECODING ----------------
def viterbi(obs_seq, model):
    """
    Viterbi algorithm: finds the most probable state sequence in log space.

    Parameters
    ----------
    obs_seq : np.ndarray, shape (T, D)
    model   : dict with keys num_states, pi, A, means, variances

    Returns
    -------
    states   : np.ndarray, shape (T,) — most probable state sequence
    log_prob : float — log-probability of the best path
    """
    N, T = model['num_states'], obs_seq.shape[0]
    log_delta = np.zeros((T, N))
    psi       = np.zeros((T, N), dtype=int)

    log_delta[0, :] = (np.log(model['pi'] + EPS)
                       + log_gaussian_prob_matrix(obs_seq[0], model['means'], model['variances']))
    for t in range(1, T):
        for j in range(N):
            seq_probs       = log_delta[t-1, :] + np.log(model['A'][:, j] + EPS)
            psi[t, j]       = np.argmax(seq_probs)
            log_delta[t, j] = np.max(seq_probs) + log_gaussian_prob(
                obs_seq[t], model['means'][j], model['variances'][j])

    # Backtrack
    states      = np.zeros(T, dtype=int)
    states[T-1] = np.argmax(log_delta[T-1, :])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, np.max(log_delta[T-1, :])


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load test descriptors and trained models
    test_desc  = np.load(TEST_DESCRIPTOR, allow_pickle=True).item()
    with open(HMM_MODEL_PATH, 'rb') as f:
        hmm_models = pickle.load(f)

    target_words        = list(hmm_models.keys())
    y_true              = []
    y_pred              = []
    recognition_results = []

    # Viterbi decoding: pick the word model with the highest log-likelihood
    for true_word, sequences in test_desc.items():
        for obs_seq in sequences:
            scores         = {word: viterbi(obs_seq, model)[1] for word, model in hmm_models.items()}
            predicted_word = max(scores, key=scores.get)
            y_true.append(true_word)
            y_pred.append(predicted_word)
            recognition_results.append((true_word, predicted_word, scores[predicted_word]))

    # Compute and display confusion matrix
    cm   = confusion_matrix(y_true, y_pred, labels=target_words)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_words)
    disp.plot(xticks_rotation=45, cmap='Blues')

    # Summary statistics
    correct    = sum(t == p for t, p, _ in recognition_results)
    total      = len(recognition_results)
    error_rate = 1 - correct / total
    print(f"\nTest set: {total} utterances | Correct: {correct} | WER: {error_rate * 100:.2f}%")

    print("\nRecognition Results (True → Predicted, Log-Likelihood):")
    for true_word, pred_word, likelihood in recognition_results:
        match = "✓" if true_word == pred_word else "✗"
        print(f"  [{match}] {true_word:<8} → {pred_word:<8}  log-likelihood: {likelihood:.2f}")
