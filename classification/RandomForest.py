import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def build_full_dataset(background_data):
    """
    Transforme background_data (structure R -> listes imbriquées)
    en un DataFrame exploitable pour la classification.
    """
    all_rows = []

    for person, pdata in background_data.items():
        for exo_name, df in pdata.items():

            if exo_name == "profile" or not isinstance(df, pd.DataFrame):
                continue

            # Sélection des features
            features = [c for c in ["ACCX","ACCY","ACCZ","ROTX","ROTY","ROTZ","ROLL","PITCH","YAW"]
                        if c in df.columns]

            # Downsample 5:1 comme en R
            df2 = df.iloc[::5].copy()
            df2 = df2[features].apply(pd.to_numeric, errors="coerce")

            # Ajouter features dérivées
            df2["ACC_NORM"] = np.sqrt(df2["ACCX"]**2 + df2["ACCY"]**2 + df2["ACCZ"]**2)
            df2["GYRO_NORM"] = np.sqrt(df2["ROTX"]**2 + df2["ROTY"]**2 + df2["ROTZ"]**2)
            df2["label"] = exo_name
            df2["person"] = person

            df2 = df2.dropna()
            all_rows.append(df2)

    if len(all_rows) == 0:
        return None

    return pd.concat(all_rows, ignore_index=True)

def random_forest_classification_full(background_data):
    """
    Équivalent du bloc R "randomForest_80/20".
    Retourne un texte formaté identique au renderPrint().
    """

    df = build_full_dataset(background_data)

    if df is None or df.empty:
        return "Dataset vide."

    # Encodage des labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    # On sépare X / y
    X = df.drop(columns=["label", "person"], errors="ignore")
    y = df["label"]

    if len(np.unique(y)) < 2:
        return "Pas assez de classes pour entraîner le modèle."

    # Train/test 80/20 équivalent à createDataPartition(stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=123, stratify=y
    )

    # Pondération automatique
    class_counts = np.bincount(y_train)
    inv = 1 / class_counts
    class_weights = inv / inv.sum()
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Random Forest équivalent R
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight=class_weight_dict,
        random_state=123
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Rapport détaillé
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    support = np.bincount(y_test)

    labels_clean = le.inverse_transform(np.arange(len(precision)))

    report = pd.DataFrame({
        "label": labels_clean,
        "precision": np.round(precision, 2),
        "recall": np.round(recall, 2),
        "f1_score": np.round(f1, 2),
        "support": support
    })

    # Macro + weighted averages
    macro = report[["precision", "recall", "f1_score"]].mean().round(2)
    weights = report["support"] / report["support"].sum()
    weighted = (report[["precision", "recall", "f1_score"]]
                .multiply(weights, axis=0)
                .sum()
                .round(2))

    accuracy = np.round((y_pred == y_test).mean(), 2)

    # Construction du texte final (comme renderPrint)
    text = "=== Classification Report (80/20) ===\n"
    text += report.to_string(index=False)
    text += f"\n\nAccuracy : {accuracy}\n"
    text += f"\nMacro avg    : {macro.tolist()}  {report['support'].sum()}"
    text += f"\nWeighted avg : {weighted.tolist()}  {report['support'].sum()}\n"

    return text

def random_forest_classification_filtered(background_data, selected_exos):
    """
    Reproduit le comportement du bloc R randomforest_results2.
    Ne garde que les exercices sélectionnés.
    """

    all_rows = []

    for person, pdata in background_data.items():
        for exo_name in selected_exos:
            if exo_name in pdata and isinstance(pdata[exo_name], pd.DataFrame):

                df = pdata[exo_name]

                features = [c for c in ["ACCX","ACCY","ACCZ","ROTX","ROTY","ROTZ","ROLL","PITCH","YAW"]
                            if c in df.columns]

                df2 = df.iloc[::5].copy()
                df2 = df2[features].apply(pd.to_numeric, errors="coerce")

                df2["ACC_NORM"] = np.sqrt(df2["ACCX"]**2 + df2["ACCY"]**2 + df2["ACCZ"]**2)
                df2["GYRO_NORM"] = np.sqrt(df2["ROTX"]**2 + df2["ROTY"]**2 + df2["ROTZ"]**2)
                df2["label"] = exo_name
                df2["person"] = person

                df2 = df2.dropna()
                all_rows.append(df2)

    if len(all_rows) == 0:
        return "Aucune donnée pour les exercices sélectionnés."

    df = pd.concat(all_rows, ignore_index=True)

    # === suite : identique à la fonction précédente ===
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    X = df.drop(columns=["label", "person"], errors="ignore")
    y = df["label"]

    if len(np.unique(y)) < 2:
        return "Pas assez de classes pour entraîner le modèle."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=123, stratify=y
    )

    class_counts = np.bincount(y_train)
    inv = 1 / class_counts
    class_weights = inv / inv.sum()
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight=class_weight_dict,
        random_state=123
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    support = np.bincount(y_test)

    labels_clean = le.inverse_transform(np.arange(len(precision)))

    report = pd.DataFrame({
        "label": labels_clean,
        "precision": np.round(precision, 2),
        "recall": np.round(recall, 2),
        "f1_score": np.round(f1, 2),
        "support": support
    })

    macro = report[["precision", "recall", "f1_score"]].mean().round(2)
    weights = report["support"] / report["support"].sum()
    weighted = (report[["precision", "recall", "f1_score"]].multiply(weights, axis=0).sum().round(2))
    accuracy = np.round((y_pred == y_test).mean(), 2)

    text = "=== Classification Report (Exos sélectionnés) ===\n"
    text += report.to_string(index=False)
    text += f"\n\nAccuracy : {accuracy}\n"
    text += f"\nMacro avg    : {macro.tolist()}  {report['support'].sum()}"
    text += f"\nWeighted avg : {weighted.tolist()}  {report['support'].sum()}\n"

    return text
