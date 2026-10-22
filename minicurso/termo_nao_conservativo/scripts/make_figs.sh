#!/bin/sh
# Roda os dois experimentos e gera as três figuras da nota (precisa de numpy e matplotlib).
set -e
cd "$(dirname "$0")"
python3 exp_energia_interna.py
python3 -W ignore exp_def42_abgrall.py
for f in fig_degrau_dirac fig_energia_interna fig_def42; do python3 -W ignore "$f.py"; done
