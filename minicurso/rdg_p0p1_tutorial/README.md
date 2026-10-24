# Tutorial rDG(P0P1) + HLLC + minmod (versão ampliada)

- `rDG_P0P1_Tutorial.tex` / `.pdf`: o tutorial (compilar com `pdflatex`, duas passagens).
- `figs/`: as sete figuras, geradas por `scripts/make_figs.sh`.
- `scripts/rdg_p0p1.py`: implementação de referência (reconstrução minmod, HLLC, BDF1/BDF2 com
  Newton–Krylov sem Jacobiana e com a iteração de ponto fixo do texto original); `python3 rdg_p0p1.py`
  roda os testes de convergência. Usa o solver de Riemann exato de `../encontro2/scripts/riemann_exato.py`.
- Requisitos: Python 3 com numpy, scipy e matplotlib.
