"""
descriptor_extraction.py
------------------------
MFCC feature extraction for isolated word recognition.

Processes a labelled audio dataset (one .mp3 file per utterance) and extracts
13-dimensional Mel-Frequency Cepstral Coefficient (MFCC) sequences for each word.
Features are saved as NumPy dictionaries keyed by word label for use in HMM training.

Module: EEEM030 – Speech & Audio Processing and Recognition
        University of Surrey
"""

import os
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ----------------
FRAME_SIZE_MS = 30      # Analysis frame length in milliseconds
HOP_SIZE_MS   = 10      # Frame hop size in milliseconds
NUM_MFCC      = 13      # Number of MFCC coefficients

# ---------------- PATHS ----------------
# Update these paths to point to your local copy of the dataset
TRAIN_PATH      = "data/DevelopmentSet"
TEST_PATH       = "data/EvaluationSet"
DESCRIPTOR_PATH = "descriptors"
os.makedirs(DESCRIPTOR_PATH, exist_ok=True)

TRAIN_DESCRIPTOR = os.path.join(DESCRIPTOR_PATH, "train_descriptors.npy")
TEST_DESCRIPTOR  = os.path.join(DESCRIPTOR_PATH, "test_descriptors.npy")

# ---------------- MFCC EXTRACTION ----------------
def extract_mfcc_from_file(file_path, sample_rate=16000):
    """
    Load an audio file and extract a sequence of MFCC feature vectors.

    Parameters
    ----------
    file_path : str
        Path to the .mp3 audio file.
    sample_rate : int
        Target sample rate in Hz. Audio is resampled if necessary.

    Returns
    -------
    np.ndarray of shape (T, NUM_MFCC)
        MFCC matrix where T is the number of frames.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    frame_length = int(FRAME_SIZE_MS * sr / 1000)
    hop_length   = int(HOP_SIZE_MS   * sr / 1000)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=NUM_MFCC,
        n_fft=frame_length, hop_length=hop_length, window="hann"
    )
    return mfcc.T  # Shape: (T, NUM_MFCC)


def process_dataset_by_word(path, target_words):
    """
    Extract MFCC features for every utterance in a dataset directory.

    Files are expected to follow the naming convention: `<id>_<word>.mp3`.
    Only files whose word label appears in `target_words` are processed.

    Parameters
    ----------
    path : str
        Directory containing .mp3 audio files.
    target_words : list of str
        Vocabulary of word labels to extract.

    Returns
    -------
    dict
        Mapping from word label (str) to a list of MFCC arrays,
        one array per utterance.
    """
    descriptors = {word: [] for word in target_words}
    for fname in os.listdir(path):
        word_label = fname.lower().split('.mp3')[0].split('_')[-1]
        if word_label in target_words:
            mfcc = extract_mfcc_from_file(os.path.join(path, fname))
            descriptors[word_label].append(mfcc)
    return descriptors


# ---------------- MAIN ----------------
if __name__ == "__main__":
    target_words = ['heed', 'hid', 'head', 'had', 'hard',
                    'hud', 'hod', 'hoard', 'hood', 'whod', 'heard']

    print("Extracting MFCC features from training set...")
    train_desc = process_dataset_by_word(TRAIN_PATH, target_words)

    print("Extracting MFCC features from test set...")
    test_desc = process_dataset_by_word(TEST_PATH, target_words)

    np.save(TRAIN_DESCRIPTOR, train_desc, allow_pickle=True)
    np.save(TEST_DESCRIPTOR,  test_desc,  allow_pickle=True)

    print(f"Done. Descriptors saved to '{DESCRIPTOR_PATH}/'")
