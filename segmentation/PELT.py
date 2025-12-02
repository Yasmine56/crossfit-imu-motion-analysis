import numpy as np
import pandas as pd
import ruptures as rpt
import plotly.express as px


# 1) Fonction : segmentation PELT + resampling

def segment_data_with_pelt_resampled(data, pen_type="MBIC", resample_factor=120):
    """
    Segmente les données accelerométriques en utilisant :
    - norme ACC_NORM
    - moyenne glissante + décimation
    - PELT (ruptures)

    Retourne : DataFrame original + colonne Segment
    """

    # 1. Calcul de la norme ACC_NORM
    acc_norm = np.sqrt(data["ACCX"]**2 + data["ACCY"]**2 + data["ACCZ"]**2)

    # 2. Rééchantillonnage : moyenne glissante + sous-échantillonnage
    roll = acc_norm.rolling(window=resample_factor, min_periods=resample_factor).mean()
    downsampled = roll.iloc[::resample_factor].dropna().values

    if len(downsampled) < 3:
        return None

    # 3. Application de PELT (ruptures)
    # Le modèle "rbf" détecte les changements de moyenne & variance -> équivalent meanvar
    algo = rpt.Pelt(model="rbf")

    try:
        # ruptures exige pen numérique : MBIC/AIC/BIC sont gérés comme chaînes
        # mais on laisse pen=pen_type pour coller à ton code R
        cps_down = algo.fit(downsampled).predict(pen=pen_type)
    except Exception:
        return None

    # ruptures ajoute souvent automatiquement le dernier point => on l'enlève
    if len(cps_down) > 0 and cps_down[-1] == len(downsampled):
        cps_down = cps_down[:-1]

    if len(cps_down) == 0:
        return None

    # 4. Conversion vers indices originaux
    cps_up = [cp * resample_factor for cp in cps_down]

    # Définition des limites des segments
    breaks = sorted(set([1] + cps_up + [len(data)]))
    breaks = [b for b in breaks if b <= len(data)]

    # 5. Attribution des segments
    segment_ids = pd.cut(
        x=range(1, len(data) + 1),
        bins=breaks,
        labels=False,
        include_lowest=True
    ) + 1  # R utilise segments commençant à 1

    data_out = data.copy()
    data_out["Segment"] = segment_ids

    return data_out


# 2) Version qui retourne data + breaks

def segment_data_with_pelt_resampled_seg(data, pen_type="MBIC", resample_factor=120):
    """
    Version qui retourne un dictionnaire :
    - "data" : DataFrame segmentée
    - "breaks" : liste des points de rupture remontés
    """

    acc_norm = np.sqrt(data["ACCX"]**2 + data["ACCY"]**2 + data["ACCZ"]**2)

    # Rééchantillonnage
    roll = acc_norm.rolling(window=resample_factor, min_periods=resample_factor).mean()
    downsampled = roll.iloc[::resample_factor].dropna().values

    if len(downsampled) < 3:
        return {"data": None, "breaks": None}

    # PELT
    algo = rpt.Pelt(model="rbf")

    try:
        cps_down = algo.fit(downsampled).predict(pen=pen_type)
    except Exception:
        return {"data": None, "breaks": None}

    if len(cps_down) > 0 and cps_down[-1] == len(downsampled):
        cps_down = cps_down[:-1]

    if len(cps_down) == 0:
        return {"data": None, "breaks": None}

    cps_up = [cp * resample_factor for cp in cps_down]

    breaks = sorted(set([1] + cps_up + [len(data)]))
    breaks = [b for b in breaks if b <= len(data)]

    segment_ids = pd.cut(
        x=range(1, len(data) + 1),
        bins=breaks,
        labels=False,
        include_lowest=True
    ) + 1

    data_out = data.copy()
    data_out["Segment"] = segment_ids

    return {"data": data_out, "breaks": breaks}
