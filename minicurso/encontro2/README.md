# Encontro 2 — Volumes finitos: upwind, Godunov e solvers de Riemann aproximados

- `Minicurso_MN_para_LC_02.tex`: slides (compilar com `lualatex`, duas passagens).
- `beamerthemeatomic.sty`: **reconstrução** do tema usado no Encontro 1 (cores, caixas
  `definicao`/`teorema`/`atencao`/`intuicao`/`mensagem`, páginas de seção, rodapé). Se o
  arquivo original do projeto estiver disponível, use-o no lugar deste (os logos da capa
  não estão incluídos aqui).
- `figs/`: figuras em PDF, geradas por `scripts/make_figs.sh`.
- `scripts/riemann_exato.py`: solver de Riemann exato para Euler 1D (Toro, cap. 4), vetorizado;
  é o arquivo "fornecido" no exercício.
- `scripts/fluxos.py`: fluxos de Godunov, Roe (com correção de Harten–Hyman ou de Harten),
  HLL, HLLC, Rusanov e Lax–Friedrichs para Burgers e Euler, com o integrador de volumes finitos
  de primeira ordem; `python3 fluxos.py` roda os testes (propriedade de Roe, Sod, problema 123).
- Requisitos: Python 3 com `numpy` e `matplotlib`; a fonte Fira Sans é opcional
  (há fallback para Latin Modern Sans).
