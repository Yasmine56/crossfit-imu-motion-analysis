import os
import json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Import de tes modules
from Classification.classif import fenetrage, extract_features_tsfel, create_segments, vue_globale
from Comptage_rep.peak_count import get_PCA, butter_lowpass, count_reps_from_dominant_freq, compute_norm
from Evaluation.evaluation_comptage import evaluate_all_exercises, evaluate_per_exercise
from Evaluation.evaluation_classif import evaluate_classification

# === Configuration ===
fs = 100  # Fréquence d’échantillonnage
EXERCICES = ["burpee", "deadlift", "front_squat", "power_clean",
             "power_snatch", "push_up", "thruster", "toes_to_bar", "wall_ball"]

# Charger le modèle Random Forest entraîné
MODEL_PATH = "Models/random_forest_model_tsfel.joblib"
MODEL = load(MODEL_PATH)

# === Chemin pour stocker les résultats ===
EVAL_PATH = "evaluation_store.json"
if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, "r") as f:
        evaluation_store = json.load(f)
else:
    evaluation_store = []

# === Vraies répétitions connues (pour évaluation) ===
true_reps_by_exercise = {ex: 10 for ex in EXERCICES}

# === Traitement de chaque exercice ===
for ex in EXERCICES:
    print(f"\n=== [Exercice: {ex}] ===")
    file_path = f"Exercices/Athlete_1/{ex}/motions.csv"
    if not os.path.exists(file_path):
        print(f"⚠️ Fichier introuvable : {file_path}")
        continue

    df = pd.read_csv(file_path)

    # --- Fenêtrage et extraction de features ---
    windows = fenetrage(df)
    features = extract_features_tsfel(windows)
    segments = create_segments(df)

    # --- Labellisation ---
    label_encoder = LabelEncoder()
    label_encoder.fit(EXERCICES)
    segments_labelised = vue_globale(segments, MODEL, label_encoder)

    # --- Comptage des répétitions pour chaque segment ---
    for segment in segments_labelised:
        label = segment['label'].iloc[0]
        data = segment[['ACCX', 'ACCY', 'ACCZ']].values
        print(f" → Traitement segment : {label}")

        # Signal brut
        clean_data = data
        expected_reps = true_reps_by_exercise.get(label, 10)

        # Méthode 1 : Norme
        norm_signal = compute_norm(clean_data)
        filtered_norm = butter_lowpass(norm_signal, cutoff=2, fs=fs)
        reps_norm = count_reps_from_dominant_freq(filtered_norm, fs)

        # Méthode 2 : PCA
        pca_signal = get_PCA(clean_data)
        filtered_pca = butter_lowpass(pca_signal, cutoff=2, fs=fs)
        reps_pca = count_reps_from_dominant_freq(filtered_pca, fs)

        # Méthode 3 : Moyenne des axes
        reps_axes = []
        for i in range(3):
            filtered_axis = butter_lowpass(clean_data[:, i], cutoff=2, fs=fs)
            reps_axes.append(count_reps_from_dominant_freq(filtered_axis, fs))
        reps_axes_mean = int(np.round(np.mean(reps_axes)))

        print(f"   → Norme : {reps_norm} | PCA : {reps_pca} | Moyenne axes : {reps_axes_mean}")

        # --- Stockage des résultats ---
        evaluation_store.append({
            'label': label,
            'reps_norm': reps_norm,
            'reps_pca': reps_pca,
            'reps_axes_mean': reps_axes_mean,
            'true_reps': expected_reps
        })

# === Sauvegarde des résultats dans JSON ===
with open(EVAL_PATH, "w") as f:
    json.dump(evaluation_store, f, indent=2)

# === Évaluation du comptage ===
evaluate_all_exercises(evaluation_store)
evaluate_per_exercise(evaluation_store)

# === Évaluation de la classification (optionnelle si tu as les vrais labels) ===
# true_labels et pred_labels à fournir si disponibles
# classes = EXERCICES
# evaluate_classification(pred_labels, true_labels, classes)
