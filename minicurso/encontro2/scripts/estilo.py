# -*- coding: utf-8 -*-
"""Estilo comum das figuras (cores do tema ATOMIC)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)

AZUL = "#103460"; LARANJA = "#ED7D31"; VERDE = "#146E46"; VERMELHO = "#AA3228"
AZULCLARO = "#5B9BD5"; CINZA = "#7F7F7F"; ROXO = "#7030A0"
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")

_fontes = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Fira Sans" if "Fira Sans" in _fontes else "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "axes.edgecolor": "0.5", "axes.linewidth": 0.7,
    "xtick.color": "0.3", "ytick.color": "0.3",
    "mathtext.fontset": "cm", "pdf.fonttype": 42,
})


def salvar(fig, nome):
    os.makedirs(FIGS, exist_ok=True)
    caminho = os.path.join(FIGS, nome)
    fig.savefig(caminho, bbox_inches="tight", pad_inches=0.03)
    print("figura gravada:", os.path.normpath(caminho))
