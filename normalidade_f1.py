import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

ALPHA = 0.05

# -----------------------------
# Helpers para detectar colunas
# -----------------------------
def pick_config_column(df: pd.DataFrame) -> str:
    """
    Tenta achar uma coluna de "configuração" por nomes comuns.
    Se não achar, usa a primeira coluna do tipo object/string.
    """
    candidates = [
        "config", "configuracao", "configuração", "configuration",
        "exp", "experimento", "experiment",
        "setup", "setting", "modelo", "model", "nome", "name"
    ]
    lowered = {c.lower(): c for c in df.columns}

    # 1) match por substring
    for key in candidates:
        for col_l, col_orig in lowered.items():
            if key in col_l:
                return col_orig

    # 2) primeira coluna não numérica
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]

    raise ValueError(
        "Não consegui detectar a coluna de configuração. "
        "Renomeie uma coluna para algo como 'config' ou passe manualmente no código."
    )

def pick_f1_column(df: pd.DataFrame) -> str:
    """
    Tenta achar coluna de F1 por nomes comuns.
    Se não achar, usa a primeira coluna numérica com valores em [0,1] (heurística).
    """
    # nomes comuns
    f1_like = ["f1", "f1_score", "f1score", "f1-score", "f1score_mean", "f1score_avg"]
    lowered = {c.lower(): c for c in df.columns}

    for key in f1_like:
        if key in lowered:
            return lowered[key]

    # substring (ex: "F1 Frontal", "metric_f1", etc.)
    for col in df.columns:
        if "f1" in col.lower():
            return col

    # heurística: numérica e em [0, 1]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if (s.min() >= 0) and (s.max() <= 1):
            return c

    raise ValueError(
        "Não consegui detectar a coluna de F1. "
        "Renomeie a coluna para conter 'f1' (ex: 'F1') ou ajuste manualmente no código."
    )

def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-_. ]+", "_", s, flags=re.UNICODE)
    s = s.strip().replace(" ", "_")
    return s[:120] if len(s) > 120 else s

# -----------------------------
# Main
# -----------------------------
def teste_normalidade(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # lê CSV de forma robusta (tenta separadores comuns)
    # se seu arquivo tem separador específico, defina: sep=";" ou sep="|"
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # fallback: tenta ';'
        df = pd.read_csv(csv_path, sep=";")

    config_col = pick_config_column(df)
    f1_col = pick_f1_column(df)

    # limpa e força numérico
    df = df.copy()
    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")

    # remove linhas sem config ou sem f1
    df = df.dropna(subset=[config_col, f1_col])

    # resumo por config
    rows = []
    configs = sorted(df[config_col].unique(), key=lambda x: str(x))

    for cfg in configs:
        x = df.loc[df[config_col] == cfg, f1_col].dropna().to_numpy(dtype=float)

        # Shapiro exige pelo menos 3 observações
        if len(x) < 3:
            rows.append({
                "config": cfg,
                "n": len(x),
                "mean_f1": np.mean(x) if len(x) else np.nan,
                "std_f1": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                "shapiro_W": np.nan,
                "shapiro_p": np.nan,
                "normal_alpha_0.05": np.nan,
                "note": "n < 3 (Shapiro não aplicável)"
            })
            continue

        W, p = stats.shapiro(x)

        # salva histograma
        plt.figure()
        plt.hist(x, bins=10)
        plt.title(f"Histograma F1 — {cfg} (n={len(x)})")
        plt.xlabel("F1")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{safe_filename(cfg)}.png"), dpi=150)
        plt.close()

        # salva Q-Q plot (sem statsmodels para manter dependências mínimas)
        # (aproximação: quantis vs quantis teóricos normais)
        plt.figure()
        (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm", plot=None)
        plt.scatter(osm, osr)
        plt.plot(osm, slope*np.array(osm) + intercept)
        plt.title(f"Q-Q plot F1 — {cfg} (Shapiro p={p:.4g})")
        plt.xlabel("Quantis teóricos (Normal)")
        plt.ylabel("Quantis observados (F1)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"qq_{safe_filename(cfg)}.png"), dpi=150)
        plt.close()

        rows.append({
            "config": cfg,
            "n": len(x),
            "mean_f1": float(np.mean(x)),
            "std_f1": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "shapiro_W": float(W),
            "shapiro_p": float(p),
            "normal_alpha_0.05": bool(p >= ALPHA),
            "note": ""
        })

    out_df = pd.DataFrame(rows).sort_values(["normal_alpha_0.05", "shapiro_p"], ascending=[True, True])
    out_path = os.path.join(out_dir, "resumo_normalidade.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    # imprime um resumo rápido no terminal
    print("Coluna de configuração detectada:", config_col)
    print("Coluna de F1 detectada:", f1_col)
    print(f"\nResumo salvo em: {out_path}\n")
    print(out_df[["config", "n", "mean_f1", "std_f1", "shapiro_p", "normal_alpha_0.05", "note"]].to_string(index=False))

