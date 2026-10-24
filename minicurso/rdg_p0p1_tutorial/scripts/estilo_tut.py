# -*- coding: utf-8 -*-
"""Estilo comum das figuras do tutorial (cores do documento)."""
import os, logging, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
logging.getLogger("fontTools").setLevel(logging.ERROR)

AZUL = "#1a5276"; AZUL2 = "#2471a3"; LARANJA = "#e67e22"; VERDE = "#27ae60"; VERMELHO = "#e74c3c"
ROXO = "#7d3c98"; CINZA = "#7f8c8d"; AMARELO = "#f1c40f"; TEAL = "#148f77"
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": AZUL, "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.framealpha": 0.92,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6, "axes.edgecolor": "0.6",
    "axes.linewidth": 0.7, "xtick.color": "0.3", "ytick.color": "0.3", "figure.dpi": 100,
    "mathtext.fontset": "dejavusans",
})


def suptitle(fig, texto):
    fig.suptitle(texto, color=AZUL, fontweight="bold", fontsize=10.5)


def salvar(fig, nome, dpi=220):
    os.makedirs(FIGS, exist_ok=True)
    caminho = os.path.join(FIGS, nome)
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print("figura gravada:", os.path.normpath(caminho))
