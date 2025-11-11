Excelente questão! O Método de Galerkin Descontínuo (DG) é muito adequado para as Equações de Euler.

A seguir, apresento (em teoria) os passos de uma discretização básica de Euler compressível 1D (escoamento com temperatura) usando o DG.

As equações de Euler compressível 1D, que descrevem o escoamento de um fluido com efeitos de temperatura (energia), são um sistema de leis de conservação hiperbólicas:

$$\frac{\partial \mathbf{u}}{\partial t} + \frac{\partial \mathbf{f}(\mathbf{u})}{\partial x} = \mathbf{0}$$

Onde:
* $\mathbf{u}$ é o vetor de **variáveis conservadas**.
* $\mathbf{f}(\mathbf{u})$ é o **fluxo**.

---

## 1. Definição das Variáveis e Equações 📝

O vetor de estado conservado $\mathbf{u}$ e o vetor de fluxo $\mathbf{f}(\mathbf{u})$ para o escoamento 1D de Euler são:

$$\mathbf{u} = \begin{pmatrix} \rho \\ \rho v \\ E \end{pmatrix}, \quad \mathbf{f}(\mathbf{u}) = \begin{pmatrix} \rho v \\ \rho v^2 + P \\ v(E + P) \end{pmatrix}$$

Onde:
* $\rho$: **Densidade**
* $v$: **Velocidade**
* $P$: **Pressão**
* $E$: **Energia total** por unidade de volume ($E = \rho e + \frac{1}{2}\rho v^2$, onde $e$ é a energia interna específica).

A **equação de estado** (por exemplo, gás ideal) relaciona $P$ às variáveis conservadas: $P = (\gamma - 1) \left(E - \frac{1}{2}\rho v^2 \right)$, onde $\gamma$ é o índice adiabático.

---

## 2. Discretização Espacial (DG) 🌐

O DG é um método de Elementos Finitos (FEM) onde a solução é **descontínua** entre os elementos (células).

### 2.1. Definição do Domínio e da Solução Aproximada

1.  **Malha (Domínio):** Dividimos o domínio 1D (o *pipe*) em $N$ elementos não sobrepostos: $\Omega = \bigcup_{k=1}^N I_k$, onde $I_k = [x_k, x_{k+1}]$.
2.  **Solução Aproximada:** Em cada elemento $I_k$, a solução $\mathbf{u}(x, t)$ é aproximada por um polinômio de grau $p$ :

    $$\mathbf{u}_h(x, t) = \sum_{j=0}^{p} \hat{\mathbf{u}}_k^j(t) \phi_j(x), \quad \forall x \in I_k$$

    Onde:
    * $\hat{\mathbf{u}}_k^j(t)$ são os coeficientes (que dependem do tempo) que precisamos resolver.
    * $\phi_j(x)$ são as funções de base local (geralmente polinômios de Legendre ou monomiais).

### 2.2. Formulação Fraca

Multiplicamos a equação de Euler pela **função de teste** $\phi_i(x)$ e integramos sobre o elemento $I_k$:

$$\int_{I_k} \phi_i \frac{\partial \mathbf{u}}{\partial t} dx + \int_{I_k} \phi_i \frac{\partial \mathbf{f}}{\partial x} dx = 0$$

Aplicamos a **integração por partes** no termo de fluxo para transferir a derivada para a função de teste:

$$\int_{I_k} \phi_i \frac{\partial \mathbf{u}}{\partial t} dx - \int_{I_k} \frac{\partial \phi_i}{\partial x} \mathbf{f}(\mathbf{u}) dx + \left[ \phi_i \mathbf{f}(\mathbf{u}) \right]_{x_k}^{x_{k+1}} = 0$$

### 2.3. Substituição do Fluxo Físico pelo Numérico (Riemman Solver)

A descontinuidade da solução $\mathbf{u}_h$ nas fronteiras dos elementos ($x_k$ e $x_{k+1}$) faz com que o fluxo $\mathbf{f}(\mathbf{u})$ seja ambíguo. Para resolver isso, o DG usa um **fluxo numérico** (também chamado de *Riemman Solver* ou **aproximador de fluxo**), $\mathbf{h}(\mathbf{u}^L, \mathbf{u}^R)$, que substitui o termo de fronteira:

$$\int_{I_k} \phi_i \frac{\partial \mathbf{u}_h}{\partial t} dx - \int_{I_k} \frac{\partial \phi_i}{\partial x} \mathbf{f}(\mathbf{u}_h) dx + \phi_i(x_{k+1}) \mathbf{h}_{k+1/2} - \phi_i(x_k) \mathbf{h}_{k-1/2} = \mathbf{0}$$

Onde:
* $\mathbf{h}_{k+1/2} = \mathbf{h}(\mathbf{u}_h(x_{k+1}^-), \mathbf{u}_h(x_{k+1}^+))$
* $\mathbf{u}_h(x_{k+1}^-)$ é a solução do lado esquerdo (dentro de $I_k$).
* $\mathbf{u}_h(x_{k+1}^+)$ é a solução do lado direito (dentro de $I_{k+1}$).

Métodos comuns para $\mathbf{h}$ incluem **Fluxo Numérico de Roe (Roe Solver)** ou **Fluxo Numérico de Harten-Lax-van Leer (HLL/HLLC)**, que são cruciais para capturar choques corretamente.

### 2.4. Sistema de Equações Ordinárias (ODE)

Substituindo a aproximação $\mathbf{u}_h$ na formulação e agrupando os termos, obtemos um grande sistema de Equações Diferenciais Ordinárias (ODEs) para os coeficientes $\hat{\mathbf{u}}_k^j(t)$:

$$\mathbf{M} \frac{d\hat{\mathbf{U}}}{dt} = \mathbf{L}(\hat{\mathbf{U}})$$

Onde:
* $\mathbf{M}$ é a **matriz de massa** (constante no tempo).
* $\hat{\mathbf{U}}$ é o vetor global de todos os coeficientes $\hat{\mathbf{u}}_k^j$.
* $\mathbf{L}(\hat{\mathbf{U}})$ é o **operador de espacial** (não linear), que inclui a integral do fluxo no volume e o termo de fluxo numérico na fronteira.

---

## 3. Discretização Temporal ⏱️

O sistema de ODEs resultante é então resolvido usando um método de integração temporal, geralmente de alta ordem (para ser consistente com a discretização DG espacial de alta ordem).

### 3.1. Escolha do Método

Para sistemas hiperbólicos, os métodos de Runge-Kutta **Totalmente Variantes na Variação (TVD)** são a escolha mais comum, pois ajudam a manter a estabilidade em problemas com choques ou gradientes acentuados.

Um esquema **Runge-Kutta de 3ª Ordem (RK3)**, por exemplo, é frequentemente utilizado:

$$\hat{\mathbf{U}}^{(1)} = \hat{\mathbf{U}}^n + \Delta t \mathbf{L}(\hat{\mathbf{U}}^n)$$
$$\hat{\mathbf{U}}^{(2)} = \frac{3}{4} \hat{\mathbf{U}}^n + \frac{1}{4} \hat{\mathbf{U}}^{(1)} + \frac{1}{4} \Delta t \mathbf{L}(\hat{\mathbf{U}}^{(1)})$$
$$\hat{\mathbf{U}}^{n+1} = \frac{1}{3} \hat{\mathbf{U}}^n + \frac{2}{3} \hat{\mathbf{U}}^{(2)} + \frac{2}{3} \Delta t \mathbf{L}(\hat{\mathbf{U}}^{(2)})$$

Onde $\Delta t$ é o passo de tempo, limitado pela condição de **CFL (Courant–Friedrichs–Lewy)** para garantir a estabilidade.

### 3.2. Passos do Algoritmo

A cada passo de tempo ($n \to n+1$):

1.  **Compute $\mathbf{L}(\hat{\mathbf{U}})$:** Calcule o lado direito do sistema de ODEs, que exige:
    * Cálculo das integrais de volume (usando quadratura de Gauss).
    * Cálculo do fluxo numérico $\mathbf{h}$ nas interfaces (usando o *Riemman Solver*).
2.  **Aplique o Esquema Temporal:** Use o resultado de $\mathbf{L}$ na iteração Runge-Kutta para avançar a solução de $\hat{\mathbf{U}}^n$ para $\hat{\mathbf{U}}^{n+1}$.

Este processo é repetido até que o tempo final desejado seja alcançado.
