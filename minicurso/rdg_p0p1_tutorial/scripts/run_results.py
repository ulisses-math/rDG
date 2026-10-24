# -*- coding: utf-8 -*-
"""Roda os casos do tutorial e guarda os resultados em results.npz (usado pelas figuras)."""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from rdg_p0p1 import *

out = {}
# --- Sod com tres malhas e dois CFL
for CFL in (1.0, 5.0):
    for N in (50, 100, 400):
        x, dx = mesh(N)
        t0 = time.time()
        U, dt, it = time_advance(sod_initial(x), dx, 0.2, CFL=CFL)
        rho, v, p = conserved_to_primitive(U)
        out[f"sod_{N}_{CFL}_x"] = x; out[f"sod_{N}_{CFL}_rho"] = rho; out[f"sod_{N}_{CFL}_v"] = v; out[f"sod_{N}_{CFL}_p"] = p
        rex = sod_exact_average(x, dx, 0.2)
        print(f"Sod CFL={CFL} N={N}: passos={len(it)}, E1(rho)={dx*np.sum(np.abs(rho-rex)):.3e}, {time.time()-t0:.1f}s")
# --- Sod com FV padrao (P0), para a comparacao direta
for N in (100, 400):
    x, dx = mesh(N)
    U, dt, it = time_advance(sod_initial(x), dx, 0.2, CFL=1.0, limiter="zero")
    rho, v, p = conserved_to_primitive(U)
    out[f"sodP0_{N}_x"] = x; out[f"sodP0_{N}_rho"] = rho; out[f"sodP0_{N}_v"] = v; out[f"sodP0_{N}_p"] = p
    print(f"Sod P0 N={N}: passos={len(it)}")
# --- convergencia em Sod: P0 vs P0P1, CFL = 1
Ns = np.array([50, 100, 200, 400, 800, 1600])
for lim in ("zero", "minmod"):
    E = []
    for N in Ns:
        x, dx = mesh(N)
        U, dt, it = time_advance(sod_initial(x), dx, 0.2, CFL=1.0, limiter=lim)
        E.append(dx * np.sum(np.abs(conserved_to_primitive(U)[0] - sod_exact_average(x, dx, 0.2))))
    out[f"conv_sod_{lim}"] = np.array(E); print("conv sod", lim, ["%.2e" % e for e in E])
out["conv_sod_N"] = Ns
# --- convergencia na onda de densidade (periodica), CFL = 1
Nw = np.array([20, 40, 80, 160, 320])
for lim in ("zero", "minmod", "none"):
    E = []
    for N in Nw:
        x, dx = mesh(N)
        U, dt, it = time_advance(density_wave_initial(x), dx, 1.0, CFL=1.0, bc="periodic", limiter=lim)
        E.append(dx * np.sum(np.abs(conserved_to_primitive(U)[0] - density_wave_exact_average(x, dx, 1.0))))
    out[f"conv_wave_{lim}"] = np.array(E); print("conv wave", lim, ["%.2e" % e for e in E])
out["conv_wave_N"] = Nw
# --- historico de Newton: ponto fixo x JFNK num passo BDF2, N=100, CFL=5 e CFL=0.5
x, dx = mesh(100); U0 = sod_initial(x)
rho, v, p = conserved_to_primitive(U0); lam = np.max(np.abs(v) + sound_speed(rho, p))
for CFL in (5.0, 0.5):
    dt = CFL * dx / lam
    U1, _ = step_newton_krylov(U0, None, dx, dt)
    _, hfp = step_fixed_point(U1, U0, dx, dt, max_iter=40)
    _, hnk = step_newton_krylov(U1, U0, dx, dt)
    out[f"newton_fp_{CFL}"] = np.array(hfp); out[f"newton_nk_{CFL}"] = np.array(hnk)
np.savez("results.npz", **out)
print("gravado results.npz")
