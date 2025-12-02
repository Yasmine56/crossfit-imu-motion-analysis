import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Évaluation globale classification
def evaluate_classification(pred_labels, true_labels, classes):
    """
    pred_labels : liste ou np.array des labels prédits
    true_labels : liste ou np.array des labels vrais
    classes     : liste des classes possibles
    """
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)

    print("\n=== Évaluation Classification Globale ===")
    acc = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy globale: {acc*100:.2f}%")

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print("\nMatrice de confusion :")
    print(df_cm)

    # Précision et rappel par classe
    precisions = precision_score(true_labels, pred_labels, labels=classes, average=None, zero_division=0)
    recalls = recall_score(true_labels, pred_labels, labels=classes, average=None, zero_division=0)
    metrics_df = pd.DataFrame({
        'Classe': classes,
        'Precision': precisions,
        'Recall': recalls
    })
    print("\nPrécision et rappel par classe :")
    print(metrics_df)
    return metrics_df
