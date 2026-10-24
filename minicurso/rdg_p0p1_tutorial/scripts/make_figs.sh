#!/bin/sh
# Gera resultados e figuras do tutorial (numpy, scipy, matplotlib).
set -e
cd "$(dirname "$0")"
python3 run_results.py
for f in fig1_malha fig2_minmod fig3_interfaces fig4_hllc fig5_tempo fig6_sod fig7_refino fig8_convergencia; do python3 "$f.py"; done
