import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix

def render_confusion_matrix(y_true, y_proba, threshold, labels=(0, 1), title="Confusion Matrix",
                            title_size=8, tick_size=7, cell_text_size=6):
    """Render confusion matrix (counts + normalized %) at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn, fp, fn, tp = cm.ravel()
    # Normalized by true class (row-normalized): recall-style normalization
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ---- Matplotlib figure
    fig, ax = plt.subplots(figsize=(2, 1.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=title_size)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=tick_size); ax.set_yticklabels(["True 0", "True 1"], fontsize=tick_size)

    # annotate counts + percentages
    for i in range(2):
        for j in range(2):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=cell_text_size)

    # Also show the raw table (optional)
    df_cm = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    return fig