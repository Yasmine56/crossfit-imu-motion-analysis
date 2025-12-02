import pandas as pd
import numpy as np



def evaluate_all_exercises(results_list):
    df = pd.DataFrame(results_list)

    print("\n=== Évaluation globale cumulée ===")
    for col in ["reps_norm", "reps_pca", "reps_axes_mean"]:
        preds = df[col].values
        true_reps = df["true_reps"].values

        mae = mean_absolute_error(true_reps, preds)
        rmse = np.sqrt(mean_squared_error(true_reps, preds))
        accuracy = 100 * np.mean(np.abs(preds - true_reps) == 0)

        print(f"\n--- {col} ---")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"Accuracy (±1 rép) : {accuracy:.1f}%")

def evaluate_per_exercise(results_list):
    df = pd.DataFrame(results_list)
    print("\n=== Évaluation par exercice ===")
    for label in df['label'].unique():
        print(f"\nExercice : {label}")
        df_label = df[df['label'] == label]
        for col in ["reps_norm", "reps_pca", "reps_axes_mean"]:
            preds = df_label[col].values
            true_reps = df_label["true_reps"].values
            mae = np.mean(np.abs(preds - true_reps) / true_reps)
            rmse = np.sqrt(np.mean(((preds - true_reps) / true_reps) ** 2))
            accuracy = 100 * (1 - np.mean(np.abs(preds - true_reps) / true_reps))
            accuracy = max(0, accuracy)
            print(f" {col} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | Acc ±1: {accuracy:.1f}%")


