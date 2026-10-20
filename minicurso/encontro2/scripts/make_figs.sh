#!/bin/sh
# Gera todas as figuras dos slides do Encontro 2 (precisa de numpy e matplotlib).
set -e
cd "$(dirname "$0")"
for f in fig_dissipacao fig_riemann_burgers fig_euler_pfunc fig_godunov_sod fig_burgers_fluxos fig_sod_fluxos fig_roe_123; do
    python3 "$f.py"
done
