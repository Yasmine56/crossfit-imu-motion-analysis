import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Évaluation globale du comptage
def evaluate_all_exercises(results_list):
    """
    results_list : liste de dicts avec keys ['label','reps_norm','reps_pca','reps_axes_mean','true_reps']
    """
    df = pd.DataFrame(results_list)
    print("\n=== Évaluation globale comptage ===")

    for col in ["reps_norm", "reps_pca", "reps_axes_mean"]:
        preds = df[col].values
        true_reps = df["true_reps"].values
        mae = mean_absolute_error(true_reps, preds)
        rmse = np.sqrt(mean_squared_error(true_reps, preds))
        accuracy = 100 * np.mean(np.abs(preds - true_reps) <= 1)

        print(f"\n--- {col} ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"Accuracy (±1 rép) : {accuracy:.1f}%")

# Évaluation par exercice
def evaluate_per_exercise(results_list):
    df = pd.DataFrame(results_list)
    print("\n=== Évaluation par exercice ===")

    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        print(f"\nExercice : {label}")

        for col in ["reps_norm", "reps_pca", "reps_axes_mean"]:
            preds = df_label[col].values
            true_reps = df_label["true_reps"].values

            mae = mean_absolute_error(true_reps, preds)
            rmse = np.sqrt(mean_squared_error(true_reps, preds))
            accuracy = 100 * np.mean(np.abs(preds - true_reps) <= 1)
            accuracy = max(accuracy, 0)

            print(f" {col} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | Acc ±1: {accuracy:.1f}%")
