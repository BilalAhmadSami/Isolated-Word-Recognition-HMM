# Isolated Word Recognition with Hidden Markov Models

**Module:** EEEM030 – Speech & Audio Processing and Recognition  
**Institution:** University of Surrey  
**Task:** Isolated word recognition over an 11-word vocabulary using Hidden Markov Models trained from scratch in NumPy

---

## Overview

This project implements a complete isolated word recognition pipeline — from raw audio to word prediction — without relying on any high-level speech or HMM toolkit (no HTK, no hmmlearn). Every component, including the forward–backward algorithm, Baum–Welch parameter estimation, and Viterbi decoding, is implemented from scratch using NumPy.

The recogniser is trained and evaluated on an 11-word vocabulary of English vowel words: **heed, hid, head, had, hard, hud, hod, hoard, hood, who'd, heard**. These words were selected deliberately for their phonetic similarity, making the discrimination task non-trivial.

---

## Pipeline

```
Raw audio (.mp3)
      │
      ▼
MFCC Feature Extraction       ← descriptor_extraction.py
      │
      ▼
HMM Training (Baum–Welch)     ← hmm_training.py
      │
      ▼
HMM Testing (Viterbi)         ← hmm_testing.py
      │
      ▼
Confusion Matrix + Results
```

---

## 1. Feature Extraction (`descriptor_extraction.py`)

MFCC (Mel-Frequency Cepstral Coefficients) features are extracted from each audio file using `librosa`:

| Parameter       | Value       |
|-----------------|-------------|
| Sample rate     | 16,000 Hz   |
| Frame size      | 30 ms       |
| Hop size        | 10 ms       |
| Window function | Hann        |
| MFCC dimensions | 13          |

Each utterance is represented as a matrix of shape `(T, 13)`, where `T` is the number of frames. Features are extracted word-by-word and saved as `.npy` files for efficient loading during training.

---

## 2. HMM Architecture (`hmm_training.py`)

Each word is modelled by an independent **left-right Hidden Markov Model** with diagonal Gaussian emission distributions.

| HMM Parameter     | Value                           |
|-------------------|---------------------------------|
| Number of states  | 8                               |
| Topology          | Left-right (no skip transitions)|
| Emission          | Diagonal Gaussian (per state)   |
| Initial state     | State 0 (deterministic, π₀ = 1)|
| Transition init   | Self-loop 0.5 / forward 0.5     |
| Emission init     | Global mean & variance          |
| Training epochs   | 15                              |

### Initialisation

All 11 HMMs are initialised with a shared global mean and variance computed over all training frames. This data-driven initialisation gives the Baum–Welch algorithm a stable starting point.

### Baum–Welch Training (EM Algorithm)

Parameters are estimated using the **batch Baum–Welch** algorithm (a specialised Expectation–Maximisation procedure for HMMs). All computations are performed in **log space** for numerical stability:

- **Forward procedure** (`log α`): computes the log probability of the partial observation sequence ending in each state at each time step.
- **Backward procedure** (`log β`): computes the log probability of the remaining observation sequence given the current state.
- **γ (gamma)**: state occupation probability — the posterior probability of being in state `i` at time `t`.
- **ξ (xi)**: state transition probability — the posterior probability of transitioning from state `i` to state `j` at time `t`.

The log-sum-exp trick is used throughout to prevent underflow when exponentiating log probabilities.

**M-step update rules:**

- `π` ← expected initial state occupancy (normalised)
- `A` ← expected transition counts (normalised row-wise)
- `μᵢ` ← γ-weighted mean of observations (per state)
- `σ²ᵢ` ← γ-weighted variance of observations (per state)

Training monitors log-likelihood and word-error rate on the training set after each epoch.

---

## 3. Viterbi Decoding and Testing (`hmm_testing.py`)

At test time, each utterance is decoded against all 11 trained HMMs using the **Viterbi algorithm**, which finds the single most probable state sequence in log space. The word whose HMM assigns the highest log-likelihood is selected as the prediction.

Recognition performance is summarised with a **confusion matrix** (visualised with `matplotlib`), which reveals which phonetically similar words are commonly confused.

---

## Repository Structure

```
Isolated-Word-Recognition-HMM/
├── descriptor_extraction.py   # MFCC feature extraction pipeline
├── hmm_training.py            # HMM initialisation + Baum–Welch training
├── hmm_testing.py             # Viterbi decoding + evaluation
├── requirements.txt
└── README.md
```

> **Note:** Pre-extracted `.npy` descriptor files and trained `.pkl` model files are not tracked in this repository (see `.gitignore`). The audio data is the EEEM030 course dataset and cannot be redistributed.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Update file paths

Each script contains a configuration section at the top. Update the path constants to point to your local copy of the audio dataset and a directory for intermediate outputs:

```python
# descriptor_extraction.py
TRAIN_PATH = "/path/to/DevelopmentSet"
TEST_PATH  = "/path/to/EvaluationSet"
DESCRIPTOR_PATH = "/path/to/Descriptors"
```

```python
# hmm_training.py / hmm_testing.py
DESCRIPTOR_PATH = "/path/to/Descriptors"
```

### 3. Extract features
```bash
python descriptor_extraction.py
```

### 4. Train HMMs
```bash
python hmm_training.py
```

### 5. Evaluate on test set
```bash
python hmm_testing.py
```

---

## Key Design Decisions

**Why implement from scratch?** The goal was to develop a deep understanding of the EM algorithm for sequential data. Using a library like `hmmlearn` would have hidden the mechanics of the forward–backward recursion and the M-step update equations.

**Why log-space arithmetic?** MFCC sequences can be 100+ frames long. Computing probabilities by multiplying 100+ small numbers in linear space causes immediate underflow to zero. Log-space addition with the log-sum-exp trick avoids this.

**Why diagonal covariance?** Full covariance matrices require estimating O(D²) parameters per state. With limited training data, diagonal approximations are more statistically efficient and avoid near-singular matrices.

---

## Tools & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| Python 3.x     | Implementation language |
| NumPy          | All HMM computations (log-domain forward/backward, Viterbi, EM updates) |
| librosa        | MFCC feature extraction from audio |
| scikit-learn   | Confusion matrix computation |
| matplotlib     | Confusion matrix visualisation |

---

## Author

**Bilal Ahmad Sami**  
MSc Artificial Intelligence, University of Surrey  
[GitHub](https://github.com/BilalAhmadSami)
