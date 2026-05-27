"""
hmm_training.py
---------------
HMM training via the Baum-Welch (EM) algorithm for isolated word recognition.

One left-right Hidden Markov Model is trained per word in the vocabulary.
All computations are performed in log space for numerical stability.
Trained models are serialised to disk for use during evaluation.

Module: EEEM030 – Speech & Audio Processing and Recognition
        University of Surrey
"""

import os
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ----------------
EPS        = 1e-6   # Small constant to prevent log(0)
NUM_STATES = 8      # Number of HMM states per word model
NUM_EPOCHS = 15     # Number of Baum-Welch training iterations

# ---------------- PATHS ----------------
DESCRIPTOR_PATH  = "descriptors"
TRAIN_DESCRIPTOR = os.path.join(DESCRIPTOR_PATH, "train_descriptors.npy")
TEST_DESCRIPTOR  = os.path.join(DESCRIPTOR_PATH, "test_descriptors.npy")
HMM_MODEL_PATH   = os.path.join(DESCRIPTOR_PATH, "hmm_models.pkl")

# ---------------- LOG-DOMAIN GAUSSIAN EMISSION ----------------
def log_gaussian_prob(x, mean, var):
    """
    Log-probability of observation vector x under a diagonal Gaussian.

    Parameters
    ----------
    x    : np.ndarray, shape (D,)
    mean : np.ndarray, shape (D,)
    var  : np.ndarray, shape (D,)

    Returns
    -------
    float : log N(x; mean, diag(var))
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


# ---------------- FORWARD / BACKWARD PROCEDURES ----------------
def forward_procedure_log(obs_seq, model):
    """
    Log-domain forward algorithm (α pass).

    Computes log α[t, j] = log P(o_1..o_t, q_t=j | model).

    Returns
    -------
    log_alpha : np.ndarray, shape (T, N)
    """
    N = model['num_states']
    T = obs_seq.shape[0]
    log_alpha = np.zeros((T, N))
    log_alpha[0, :] = (np.log(model['pi'] + EPS)
                       + log_gaussian_prob_matrix(obs_seq[0], model['means'], model['variances']))
    for t in range(1, T):
        for j in range(N):
            temp = log_alpha[t-1, :] + np.log(model['A'][:, j] + EPS)
            log_alpha[t, j] = log_sum_exp(temp) + log_gaussian_prob(
                obs_seq[t], model['means'][j], model['variances'][j])
    return log_alpha


def backward_procedure_log(obs_seq, model):
    """
    Log-domain backward algorithm (β pass).

    Computes log β[t, i] = log P(o_{t+1}..o_T | q_t=i, model).

    Returns
    -------
    log_beta : np.ndarray, shape (T, N)
    """
    N = model['num_states']
    T = obs_seq.shape[0]
    log_beta = np.zeros((T, N))  # log β[T-1, :] = log(1) = 0
    for t in range(T - 2, -1, -1):
        for i in range(N):
            temp = (np.log(model['A'][i, :] + EPS)
                    + log_gaussian_prob_matrix(obs_seq[t+1], model['means'], model['variances'])
                    + log_beta[t+1, :])
            log_beta[t, i] = log_sum_exp(temp)
    return log_beta


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
    states     : np.ndarray, shape (T,) — most probable state sequence
    log_prob   : float — log-probability of the best path
    """
    N, T = model['num_states'], obs_seq.shape[0]
    log_delta = np.zeros((T, N))
    psi       = np.zeros((T, N), dtype=int)

    log_delta[0, :] = (np.log(model['pi'] + EPS)
                       + log_gaussian_prob_matrix(obs_seq[0], model['means'], model['variances']))
    for t in range(1, T):
        for j in range(N):
            seq_probs     = log_delta[t-1, :] + np.log(model['A'][:, j] + EPS)
            psi[t, j]     = np.argmax(seq_probs)
            log_delta[t, j] = np.max(seq_probs) + log_gaussian_prob(
                obs_seq[t], model['means'][j], model['variances'][j])

    # Backtrack
    states       = np.zeros(T, dtype=int)
    states[T-1]  = np.argmax(log_delta[T-1, :])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, np.max(log_delta[T-1, :])


# ---------------- HMM INITIALISATION ----------------
def compute_global_mean_variance(descriptor_dict):
    """
    Compute the global mean and variance across all training frames.
    Used to initialise HMM emission parameters.
    """
    all_frames = np.vstack([mfcc for seq_list in descriptor_dict.values() for mfcc in seq_list])
    return np.mean(all_frames, axis=0), np.var(all_frames, axis=0) + EPS


def initialize_hmm(target_words, num_states, global_mean, global_variance):
    """
    Initialise one left-right HMM per word.

    Topology: each state transitions to itself (self-loop) or the next state
    with equal probability (0.5 / 0.5), except the final state which is absorbing.
    All emission Gaussians are initialised to the corpus-wide mean and variance.

    Returns
    -------
    dict mapping word (str) → model (dict)
    """
    hmm_models = {}
    for word in target_words:
        model = {
            'num_states': num_states,
            'pi':         np.zeros(num_states),
            'A':          np.zeros((num_states, num_states)),
            'means':      np.tile(global_mean,     (num_states, 1)),
            'variances':  np.tile(global_variance, (num_states, 1))
        }
        model['pi'][0] = 1.0  # Always start in state 0
        for i in range(num_states):
            if i < num_states - 1:
                model['A'][i, i]     = 0.5
                model['A'][i, i + 1] = 0.5
            else:
                model['A'][i, i] = 1.0  # Absorbing final state
        hmm_models[word] = model
    return hmm_models


# ---------------- TRAINING EVALUATION ----------------
def evaluate_training(train_desc, hmm_models):
    """
    Compute word-error rate and average Viterbi log-likelihood on the training set.
    Used to monitor convergence across epochs.
    """
    total_files  = 0
    correct      = 0
    total_score  = 0.0
    for true_word, sequences in train_desc.items():
        for obs_seq in sequences:
            scores         = {word: viterbi(obs_seq, model)[1] for word, model in hmm_models.items()}
            predicted_word = max(scores, key=scores.get)
            total_score   += scores[predicted_word]
            total_files   += 1
            if predicted_word == true_word:
                correct += 1
    error_rate = 1 - (correct / total_files)
    avg_score  = total_score / total_files
    return avg_score, error_rate


# ---------------- BATCH BAUM-WELCH ----------------
def batch_baum_welch_update(sequences, model):
    """
    One full Baum-Welch (EM) update over all training sequences for a single word model.

    E-step: compute γ (state occupation) and ξ (transition) posteriors via
            the log-domain forward-backward algorithm.
    M-step: re-estimate π, A, emission means, and emission variances from
            the accumulated sufficient statistics.

    Parameters
    ----------
    sequences : list of np.ndarray, each shape (T_i, D)
    model     : dict (updated in-place)

    Returns
    -------
    float : total log-likelihood over all sequences
    """
    N = model['num_states']

    # Sufficient statistics accumulators
    acc_pi    = np.zeros(N)
    acc_A     = np.zeros((N, N))
    acc_means = np.zeros_like(model['means'])
    acc_vars  = np.zeros_like(model['variances'])
    total_gamma = np.zeros(N)
    total_log_likelihood = 0.0

    for obs_seq in sequences:
        T = obs_seq.shape[0]
        log_alpha    = forward_procedure_log(obs_seq, model)
        log_beta     = backward_procedure_log(obs_seq, model)
        log_likelihood = log_sum_exp(log_alpha[T-1, :])
        total_log_likelihood += log_likelihood

        # γ: state occupation posteriors
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma     = np.exp(np.clip(log_gamma, -700, 700))
        gamma    /= np.sum(gamma, axis=1, keepdims=True)

        # ξ: transition posteriors
        xi = np.zeros((T-1, N, N))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (log_alpha[t, i]
                                   + np.log(model['A'][i, j] + EPS)
                                   + log_gaussian_prob(obs_seq[t+1], model['means'][j], model['variances'][j])
                                   + log_beta[t+1, j])
            xi[t]  = np.exp(np.clip(xi[t] - log_sum_exp(xi[t].flatten()), -700, 700))
            xi[t] /= np.sum(xi[t])

        # Accumulate sufficient statistics
        acc_pi += gamma[0]
        acc_A  += np.sum(xi, axis=0)
        for i in range(N):
            gamma_i      = gamma[:, i][:, None]
            acc_means[i] += np.sum(gamma_i * obs_seq, axis=0)
            acc_vars[i]  += np.sum(gamma_i * (obs_seq - model['means'][i]) ** 2, axis=0)
            total_gamma[i] += np.sum(gamma_i)

    # M-step: update parameters
    model['pi'] = acc_pi / np.sum(acc_pi)
    model['A']  = acc_A + EPS
    model['A'] /= np.sum(model['A'], axis=1, keepdims=True)
    for i in range(N):
        model['means'][i]     = acc_means[i] / total_gamma[i]
        model['variances'][i] = acc_vars[i]  / total_gamma[i] + EPS

    return total_log_likelihood


# ---------------- MAIN ----------------
if __name__ == "__main__":
    target_words = ['heed', 'hid', 'head', 'had', 'hard',
                    'hud', 'hod', 'hoard', 'hood', 'whod', 'heard']

    # Load pre-extracted MFCC descriptors
    train_desc = np.load(TRAIN_DESCRIPTOR, allow_pickle=True).item()
    test_desc  = np.load(TEST_DESCRIPTOR,  allow_pickle=True).item()

    # Initialise HMMs from corpus-wide statistics
    mean, variance = compute_global_mean_variance(train_desc)
    hmm_models     = initialize_hmm(target_words, NUM_STATES, mean, variance)

    # Baum-Welch training loop
    print(f"Training {len(target_words)} HMMs for {NUM_EPOCHS} epochs...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        total_log_likelihood = 0.0
        for word, sequences in train_desc.items():
            total_log_likelihood += batch_baum_welch_update(sequences, hmm_models[word])

        avg_score, error_rate = evaluate_training(train_desc, hmm_models)
        print(f"Epoch {epoch:02d} | Log-Likelihood: {total_log_likelihood:.2f} | "
              f"Avg Score: {avg_score:.3f} | Train WER: {error_rate * 100:.2f}%")

    # Serialise trained models
    with open(HMM_MODEL_PATH, 'wb') as f:
        pickle.dump(hmm_models, f)
    print(f"\nTrained models saved to '{HMM_MODEL_PATH}'")
