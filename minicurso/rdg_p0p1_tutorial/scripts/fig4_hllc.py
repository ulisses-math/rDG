# -*- coding: utf-8 -*-
"""Figura 4: Passo 3, o fluxo HLLC: ondas, os quatro ramos e o mapa de regimes."""
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import *

fig = plt.figure(figsize=(12.5, 7.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.5, wspace=0.35)
# (a) diagrama de ondas para a interface central de Sod
a = fig.add_subplot(gs[0, :2])
UL = primitive_to_conserved(*[np.array([v]) for v in SOD_L]); UR = primitive_to_conserved(*[np.array([v]) for v in SOD_R])
F, reg, (SL, Ss, SR) = hllc_flux(UL, UR)
SL, Ss, SR = float(SL[0]), float(Ss[0]), float(SR[0])
T = 1.0
a.fill_betweenx([0, T], [-2.2, -2.2], [0, SL * T], color=AZUL2, alpha=0.12)
a.fill_betweenx([0, T], [0, SL * T], [0, Ss * T], color=VERDE, alpha=0.15)
a.fill_betweenx([0, T], [0, Ss * T], [0, SR * T], color=LARANJA, alpha=0.15)
a.fill_betweenx([0, T], [0, SR * T], [2.6, 2.6], color=VERMELHO, alpha=0.10)
a.plot([0, SL * T], [0, T], color=AZUL, lw=2.2, label=r"$S_L=%.2f$" % SL)
a.plot([0, Ss * T], [0, T], color=VERDE, lw=2.2, ls="--", label=r"$S_*=%.2f$ (contato)" % Ss)
a.plot([0, SR * T], [0, T], color=VERMELHO, lw=2.2, label=r"$S_R=%.2f$" % SR)
a.axvline(0, color="k", lw=1.2, ls=":", label=r"a interface, $x/t=0$")
for xx_, yy_, lab in ((-1.6, 0.6, r"$\mathbf{u}_L$"), (-0.3, 0.6, r"$\mathbf{u}_{*L}$"), (0.72, 0.45, r"$\mathbf{u}_{*R}$"), (2.0, 0.45, r"$\mathbf{u}_R$")):
    a.text(xx_, yy_, lab, ha="center", fontsize=11)
a.text(0.05, 0.08, "aqui: $S_L<0<S_*$\n⇒ ramo $\\mathbf{f}_{*L}$", fontsize=8, color=VERDE, ha="left")
a.set_xlim(-2.2, 2.6); a.set_ylim(0, T * 1.05); a.set_xlabel("x"); a.set_ylabel("t")
a.set_title("(a) As três ondas na interface central de Sod")
a.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 1.0))
# (b) os quatro ramos
casos = [("(i) $S_L>0$: tudo vai para a direita", r"$\hat f=\mathbf{f}_L$", (0.6, 1.4, 2.2)),
         ("(ii) $S_L\\leq 0<S_*$", r"$\hat f=\mathbf{f}_{*L}$", (-1.5, 0.6, 2.0)),
         ("(iii) $S_*\\leq 0\\leq S_R$", r"$\hat f=\mathbf{f}_{*R}$", (-2.0, -0.6, 1.5)),
         ("(iv) $S_R<0$: tudo vai para a esquerda", r"$\mathbf{f}_R$", (-2.2, -1.4, -0.6))]
for k, (tit, lab, (sl, ss, sr)) in enumerate(casos):
    b = fig.add_subplot(gs[1, k])
    b.fill_betweenx([0, 1], [-2.5, -2.5], [0, sl], color=AZUL2, alpha=0.12)
    b.fill_betweenx([0, 1], [0, sl], [0, ss], color=VERDE, alpha=0.15)
    b.fill_betweenx([0, 1], [0, ss], [0, sr], color=LARANJA, alpha=0.15)
    b.fill_betweenx([0, 1], [0, sr], [2.5, 2.5], color=VERMELHO, alpha=0.10)
    b.plot([0, sl], [0, 1], color=AZUL, lw=1.8); b.plot([0, ss], [0, 1], color=VERDE, lw=1.8, ls="--"); b.plot([0, sr], [0, 1], color=VERMELHO, lw=1.8)
    b.axvline(0, color="k", lw=1.4, ls=":")
    b.set_xlim(-2.5, 2.5); b.set_ylim(0, 1.1); b.set_xticks([]); b.set_yticks([])
    b.set_title(tit, fontsize=8.5)
    b.text(0, 1.0, lab.replace(r"$\mathbf{f}_R$", r"$\hat f=\mathbf{f}_R$"), ha="center", va="top", fontsize=10,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
    b.set_xlabel("x"); 
    if k == 0: b.set_ylabel("t")
# (c) mapa de regimes num escoamento com velocidade de fundo
c = fig.add_subplot(gs[0, 2:])
N = 100; x, dx = mesh(N)
v0 = -0.5
U0 = primitive_to_conserved(np.where(x < 0.5, 1.0, 0.125), v0 + 0 * x, np.where(x < 0.5, 1.0, 0.1))
U, dt, it = time_advance(U0, dx, 0.1, CFL=1.0)
S = reconstruct(U); uL, uR = interface_states(U, S)
F, reg, _ = hllc_flux(uL, uR)
rho = conserved_to_primitive(U)[0]
xf = np.concatenate(([0.0], x[:-1] + dx / 2, [1.0]))
c.plot(x, rho, color="0.3", lw=1.2, label=r"$\rho$ em $t=0{,}1$ (N = 100)")
cores = {0: AZUL, 1: VERDE, 2: LARANJA, 3: VERMELHO}; nomes = {0: r"ramo (i) $\mathbf{f}_L$", 1: r"ramo (ii) $\mathbf{f}_{*L}$", 2: r"ramo (iii) $\mathbf{f}_{*R}$", 3: r"ramo (iv) $\mathbf{f}_R$"}
for r in (0, 1, 2, 3):
    m = reg == r
    if m.any():
        c.plot(xf[m], np.interp(xf[m], x, rho), "|", color=cores[r], ms=14, mew=2.2, label=nomes[r])
c.set_xlim(0, 1); c.set_ylim(0, 1.15); c.set_xlabel("x"); c.set_ylabel(r"$\rho$")
c.set_title(r"(c) Mapa de regimes: Sod com velocidade de fundo $v_0=-0{,}5$")
c.legend(loc="upper right", fontsize=7.5)
suptitle(fig, "Passo 3 — O fluxo numérico HLLC")
salvar(fig, "fig4_hllc.png")
