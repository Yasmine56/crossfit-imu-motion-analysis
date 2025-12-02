import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

# Norme du signal
def compute_norm(data):
    """
    Calcul de la norme du signal accelerometrique.
    data : np.array shape (n_samples, 3) ACCX, ACCY, ACCZ
    """
    return np.linalg.norm(data, axis=1)

# Filtre passe-bas
def butter_lowpass(signal, cutoff=2, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# PCA sur le signal
def get_PCA(data):
    """
    Retourne le premier composant principal du signal 3 axes
    """
    pca = PCA(n_components=1)
    return pca.fit_transform(data).flatten()

# Comptage à partir de la fréquence dominante
def count_reps_from_dominant_freq(signal, fs=100):
    """
    Estimation du nombre de répétitions à partir de la fréquence dominante
    """
    from scipy.fft import rfft, rfftfreq

    n = len(signal)
    yf = np.abs(rfft(signal - np.mean(signal)))
    xf = rfftfreq(n, 1/fs)
    idx = np.argmax(yf[1:]) + 1  # Ignorer DC
    freq_dom = xf[idx]
    duration = n / fs
    reps_estimated = int(round(freq_dom * duration))
    return reps_estimated
