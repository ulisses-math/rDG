# -*- coding: utf-8 -*-
"""Figura 5: Passo 4, balanco de fluxos, estencil do BDF2 e convergencia de Newton."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import *

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"))
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), gridspec_kw=dict(width_ratios=[1.15, 1, 1.1]))
# (a) balanco numa celula perto do choque
N = 50; x, dx = mesh(N)
U, dt, it = time_advance(sod_initial(x), dx, 0.1, CFL=1.0)
S = reconstruct(U); uL, uR = interface_states(U, S); F, _, _ = hllc_flux(uL, uR)
rho = conserved_to_primitive(U)[0]
i = int(np.argmax(np.abs(np.diff(rho)))) + 0          # celula imediatamente a esquerda do maior salto
i = max(i, 1)
a = axs[0]
a.bar(x[i - 1:i + 2], rho[i - 1:i + 2], width=dx, color=[ "0.85", AZUL2, "0.85"], edgecolor="0.5")
fin, fout = F[0, i], F[0, i + 1]
a.annotate("", xy=(x[i] - dx / 2 - 0.002, 0.55), xytext=(x[i] - 1.1 * dx, 0.55), arrowprops=dict(arrowstyle="-|>", color=VERDE, lw=2))
a.text(x[i] - 0.8 * dx, 0.62, r"entra: $\hat f_{i-1/2}=%.3f$" % fin, color=VERDE, fontsize=8, ha="center", bbox=dict(fc="white", ec="none", pad=1))
a.annotate("", xy=(x[i] + 1.1 * dx, 0.55), xytext=(x[i] + dx / 2 + 0.002, 0.55), arrowprops=dict(arrowstyle="-|>", color=VERMELHO, lw=2))
a.text(x[i] + 0.8 * dx, 0.62, r"sai: $\hat f_{i+1/2}=%.3f$" % fout, color=VERMELHO, fontsize=8, ha="center", bbox=dict(fc="white", ec="none", pad=1))
a.text(x[i], rho[i] + 0.06, r"$\bar\rho_i=%.3f$" % rho[i], ha="center", fontsize=8.5, color=AZUL)
a.text(x[i], max(rho[i - 1:i + 2]) * 1.42, r"$\frac{d\bar\rho_i}{dt}=-\frac{%.3f-%.3f}{\Delta x}=%+.2f$" % (fout, fin, -(fout - fin) / dx), ha="center", fontsize=8.5,
       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
a.set_xlim(x[i] - 1.6 * dx, x[i] + 1.6 * dx); a.set_ylim(0, max(rho[i - 1:i + 2]) * 1.65)
a.set_xticks([x[i] - dx / 2, x[i], x[i] + dx / 2]); a.set_xticklabels([r"$x_{i-1/2}$", r"$x_i$", r"$x_{i+1/2}$"])
a.set_title("(a) Balanço de massa numa célula (perto do choque)"); a.set_ylabel(r"$\rho$")
# (b) estencil do BDF2
b = axs[1]
for k, (t, lab, w) in enumerate((( -1, r"$\bar u^{\,n-1}$", "+1"), (0, r"$\bar u^{\,n}$", "−4"), (1, r"$\bar u^{\,n+1}$", "+3"))):
    cor = CINZA if k < 2 else LARANJA
    b.plot([t], [0], "o", color=cor, ms=14)
    b.text(t, -0.32, lab, ha="center", fontsize=10)
    b.text(t, 0.3, "peso " + w, ha="center", fontsize=9, color=cor)
b.annotate("", xy=(1, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle="-", color="0.6", lw=1.2))
b.text(0, 0.85, r"$3\bar u^{\,n+1}-4\bar u^{\,n}+\bar u^{\,n-1}=2\Delta t\,L(\bar u^{\,n+1})$", ha="center", fontsize=9.5,
       bbox=dict(boxstyle="round,pad=0.3", fc="#fcf3cf", ec=LARANJA))
b.text(1, -0.75, "incógnita: aparece\ndentro de $L$ também\n(sistema não linear)", ha="center", fontsize=8, color=LARANJA)
b.text(-0.5, -0.75, "conhecidos", ha="center", fontsize=8, color=CINZA)
b.set_xlim(-1.7, 1.7); b.set_ylim(-1.1, 1.2); b.set_xticks([-1, 0, 1]); b.set_xticklabels([r"$t^{n-1}$", r"$t^n$", r"$t^{n+1}$"]); b.set_yticks([])
b.grid(False); b.set_title("(b) O estêncil do BDF2 no tempo")
# (c) convergencia: ponto fixo x JFNK
c = axs[2]
for CFL, ls in ((5.0, "-"), (0.5, "--")):
    hfp = d[f"newton_fp_{CFL}"]; hnk = d[f"newton_nk_{CFL}"]
    c.semilogy(np.arange(1, len(hfp) + 1), hfp, ls=ls, color=VERMELHO, marker="s", ms=3, label=f"ponto fixo, CFL = {CFL:g}")
    c.semilogy(np.arange(1, len(hnk) + 1), hnk, ls=ls, color=AZUL, marker="o", ms=3, label=f"Newton–Krylov, CFL = {CFL:g}")
c.set_ylim(1e-12, 1e8); c.set_xlim(0, 25); c.set_xlabel("iteração"); c.set_ylabel(r"$\max|\mathbf{F}|$ (resíduo)")
c.set_title("(c) Resolvendo o sistema não linear (N = 100)")
c.legend(loc="upper right", fontsize=7)
suptitle(fig, "Passo 4 — Atualização temporal")
fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
salvar(fig, "fig5_tempo.png")
