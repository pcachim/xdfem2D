# Validação do motor de cálculo

Conjunto de estruturas simples com solução analítica fechada, para validar o
motor de elementos finitos (barra N/V/M e triângulo CST em estado plano de
tensão). Espelha o documento `Validacao_Programa_MEF.docx`.

## Conteúdo

| Ficheiro | Função |
|---|---|
| `validation_cases.py` | Fonte única (estática): constrói os modelos, guarda os valores analíticos e os extractores de resultados. |
| `modal_cases.py` | O mesmo para a análise modal e o espectro de resposta. |
| `build_x2d.py` | Gera os `.x2d` em `models/` (um por caso; o 3.2 em 3 malhas). |
| `make_report.py` | Corre os modelos, preenche as colunas *Resultado MEF* / *Erro (%)* dos capítulos do template, **gera** os capítulos novos e reestrutura o documento em partes. |
| `report_sections.py` | Texto, soluções analíticas e extractores dos capítulos gerados (arcos, Allman, molas, dinâmica). |
| `models/*.x2d` | Os modelos de validação (estrutura, sem resultados). |
| `validate_quads.py` | Validação dos elementos quad (QM6/MITC4): patch de tensão constante, laje SS vs. Timoshenko, e consola em flexão pura vs. Euler. Corre e imprime PASS/FAIL (`python validation/validate_quads.py`). |
| `Validacao_Programa_MEF.docx` | Template do documento. |
| `Validacao_Programa_MEF_resultados.docx` | Documento preenchido (gerado). |

Os elementos **quad** têm também exemplo trabalhado em
[`examples/example-slab-quad.py`](../examples/example-slab-quad.py) (a mesma
laje simplesmente apoiada meshada com triângulos MITC3 e com quads MITC4, na
mesma grelha, a mostrar que os quads usam metade dos elementos e eliminam o
viés diagonal do campo de momentos) e cobertura de regressão em
`tests/tests_engine/test_quad_elements*.py` e `tests/tests_model/test_quad_dynamics.py`.

O teste automático vive em `tests/tests_model/test_validation_x2d.py`.

## Como usar

```bash
python validation/build_x2d.py          # (re)gerar os .x2d
python -m pytest tests/tests_model/test_validation_x2d.py   # validar
python validation/make_report.py        # gerar o documento preenchido
```

Unidades consistentes: m, kN, kN/m² (÷1000 → MPa), logo E = 30×10⁶ kN/m².

O documento gerado organiza-se em seis partes e nove capítulos:

| Parte | Capítulos | Origem |
|---|---|---|
| — | 1. Objetivo e metodologia | template |
| I — Elementos de barra | 2. Barras (casos 2.1–2.5) · 3. Estruturas de eixo curvo (arco) | template · gerado |
| II — Elementos triangulares | 4. CST (casos 4.1–4.4) · 5. Elemento de Allman | template · gerado |
| III — Elementos de placa e grelha | 6. Placa DKT e MITC3 (P.1–P.2) · 7. Grelha (G.1–G.3) | gerado |
| IV — Molas | 8. Molas de nó e de elemento | gerado |
| V — Análise dinâmica | 9. Modal e espectro de resposta | gerado |
| VI — Ligações (constraints) | 10. Ligações multiponto entre graus de liberdade | gerado |
| VII — Conclusões | 11. Quadro-resumo e recomendações | template |

Parte III — flexão fora do plano. Placa: **P.1** valida a laje fina com o DKT
contra a série de Navier (Kirchhoff) e **P.2** a laje espessa com o MITC3 contra
a série exacta de Mindlin (deformação por corte) — a flecha que o elemento fino
não representa. Grelha (barras de placa): **G.1** flexão da consola (FL³/3EI,
FL²/2EI), **G.2** torção de St-Venant (TL/GJ) e **G.3** grelha de duas vigas
cruzadas (repartição da carga) — todas em fórmula fechada exacta.

O `make_report.py` reestrutura o template ao gerar: promove os rótulos «Parte»
a títulos próprios, renumera os capítulos (o CST passa de 3 para 4, e com ele os
rótulos dos casos 3.x → 4.x e as referências no texto), insere os capítulos
gerados nas posições certas e deixa o quadro-resumo no fim. O quadro-resumo
recebe automaticamente as linhas dos capítulos gerados, por ordem de capítulo.

Os capítulos gerados escrevem-se a partir de `report_sections.py` — título,
descrição, solução analítica, tabela e notas — pelo que o template fica intacto
e não há texto duplicado: os números citados na prosa saem das mesmas constantes
que os testes usam. O índice do template é um campo estático e não lista os
capítulos gerados (basta actualizá-lo no Word com Ctrl+A, F9).

## Os casos

Barra (2.1–2.5) e CST (3.1–3.4) — 9 casos, cobrindo carga pontual,
distribuída, assentamento e temperatura. Todos reproduzidos exatamente
(erro < 10⁻³ %), exceto:

- **3.2 (consola-parede CST):** o alvo 17,26 mm vem da teoria de vigas, não da
  elasticidade plana; o CST converge lentamente por baixo. Valida-se a
  *convergência monótona* em 3 malhas (16×4 → 32×8 → 64×16 = 16,77 mm, −2,8 %),
  não um valor exato. O pico de τ_xy é uma concentração de canto, não comparável
  ponto-a-ponto com 1,5·V/A.
- **3.3 (deslocamento imposto):** o apoio do documento (bordo x=0 totalmente
  encastrado) impede a contração de Poisson e o estado deixa de ser uniforme.
  Aqui o bordo x=0 restringe apenas u_x (um canto fixo em u_y), tornando-o o
  patch uniaxial pretendido — σ_x = 15 MPa e reação 1500 kN exatos em qualquer
  malha.

Cada caso inclui ainda a verificação universal de equilíbrio global
(ΣReações + ΣCargas ≈ 0).

## Casos extra

Verificações adicionais pedidas depois. Entram todas na suite de testes; o arco
(capítulo 3) e os casos de Allman (capítulo 5) entram também no documento
gerado, enquanto os casos por objeto de geometria (2.1a, 3.1a) ficam só nos
testes, por duplicarem casos já documentados.

**Nota sobre numeração:** os identificadores dos modelos (`3.1.x2d`, `3.2-32x8`,
`s-a1`, `m-b1-16`, …) são estáveis e independentes do documento. A numeração dos
capítulos e dos casos no `.docx` é atribuída na geração, pelo que o caso `3.1`
dos modelos aparece no documento como Caso 4.1 (o CST passou a ser o capítulo 4).

- **2.1a — viga por objeto-linha.** O caso 2.1 construído como uma polilinha
  (0,0)-(3,0)-(6,0) que se discretiza em barras, com o meio-vão como nó de
  definição para receber a carga. Reproduz 2.1 exatamente (M=75, V=25, δ=2,40).
- **3.1a — placa por objeto-retângulo.** O caso 3.1 como retângulo malhado em
  CST, com tração aplicada por carga de bordo de superfície. σ_x=1,0 MPa e
  δ_x=0,0667 mm exatos em qualquer malha. **Nota:** a propagação de apoios ao
  longo de um bordo só ocorre quando *ambos os cantos têm o mesmo apoio* — por
  isso os dois cantos esquerdos levam o mesmo apoio ux (propaga) e o u_y fica
  num ponto em y=0 (consistente com o campo exato).
- **arco — consola em quarto de círculo.** Raio R=3 m, carga P=10 kN na ponta
  livre. Solução analítica (flexão, Castigliano) δ_v = πPR³/(4EI) = 2,262 mm; o
  FE (cordas retas + axial/corte) converge por cima (div 16→64: +0,03 %→+0,22 %).
  Momento no encastramento (30 kN·m) e reação vertical (10 kN) são exatos.
- **Allman — patch test canónico + 3.1/3.2/3.3.** O triângulo de Allman tem um
  grau de liberdade de *drilling* (rotação tz) por nó:
  - **Patch test de Allman (1984)** — `tests/tests_model/test_allman_patch.py`.
    No quadrado com 4 triângulos e **nó interior livre**, com a fronteira
    prescrita (u, v **e θ**), o elemento reproduz **exatamente** (≤10⁻¹²):
    translação rígida, rotação rígida, e deformação constante uniaxial, biaxial
    e de corte puro. É o teste de referência e o elemento **passa**.
  - **3.1-allman e 3.3-allman:** os mesmos patches por via mecânica (carga de
    bordo / deslocamento imposto), com o drilling restringido — exatos.
  - **3.2-allman (flexão):** com o drilling livre, converge para a teoria de
    vigas **mais depressa que o CST** (32×8: −1,6 % vs −10 %). É o ganho do
    elemento.
  - **3.4-allman (térmico):** a carga térmica e a recuperação de tensões passam
    a cobrir o Allman (f_th = t·A·B0ᵀ·D·ε₀ nos 9 DOF; σ = D·(B·u − ε₀)). Expansão
    livre → σ=0, δx=0,60, δy=0,30 (exato); restrição total → σ = −D·ε₀.

  Nota: prescrever θ na fronteira faz parte do patch test de um elemento com
  drilling; deixar θ livre na fronteira é um patch mal-posto, não uma falha do
  elemento. Em análise real o drilling fica livre e funciona (ver 3.2-allman).

## Molas (apoios elásticos e fundação de Winkler)

Casos `s-*`, gerados no documento (Parte III). O teste automático vive em
`tests/tests_model/test_springs_validation.py` (os casos lineares exatos entram
também na suite genérica de `test_validation_x2d.py`).

Duas naturezas distintas, e é isso que determina o desenho dos casos:

- **Mola nodal** (`kx, ky, kt`) entra na diagonal de K. Toda a solução fechada
  de "estrutura + apoio elástico" é reproduzida **exatamente** (erro ≤ 10⁻¹² %).
- **Mola de elemento** (Winkler) é agregada de forma ***lumped*** — `k·L/2` em
  cada nó extremo — e não como matriz de fundação consistente. Logo: um estado
  **uniforme** continua exato (a resultante é conservada), mas um estado
  **variável** converge apenas em O(h²), e por isso valida-se por convergência.

| Caso | Solução teórica |
|---|---|
| `s-a1` | Mola em paralelo com a barra: δ = P/(EA/L + k), N = P·(EA/L)/(EA/L + k) |
| `s-a2` | Consola com apoio elástico: δ = P/(k + 3EI/L³), M = (P − kδ)·L |
| `s-a3` | Molas de rotação nos apoios: M_apoio = qL²/12 · 1/(1 + 2EI/ktL) |
| `s-a4` | Corpo rígido só sobre molas (sem apoios): R por estática, w = R/k |
| `s-b1` | Bloco rígido sobre Winkler: w = q/k, qualquer malha |
| `s-b5` | Mola de elemento em eixos locais numa barra a 45°: G = R·diag(ka,kt)·Rᵀ |
| `s-b2` | Viga longa sobre fundação (Hetényi): w₀ = Pλ/2k, M₀ = P/4λ |
| `s-b3` | Viga semi-infinita, carga no bordo: w₀ = 2Pλ/k, M_max = 0,3224 P/λ |
| `s-c1`, `s-c2` | Molas unilaterais: activa → resultado bilateral; inactiva → δ = PL³/3EI |
| `s-c3` | Sapata rígida com contacto parcial: a = 3(L/2 − e), p_max = 2N/a |

Notas:

- **`s-b1` — artefacto do lumping.** Como a reação da fundação é concentrada nos
  nós, o elemento "vê" a carga distribuída desequilibrada e aparece um momento
  espúrio de `q·h²/8`. O assentamento é exato; o momento é verificado a decair
  exatamente para 1/4 a cada refinamento (O(h²)), o que documenta o erro em vez
  de o esconder.
- **`s-b2` / `s-b3` — λ = (k/4EI)^(1/4).** Com k = 4×10⁴ kN/m², λ = 0,571 m⁻¹ e
  L = 20 m (λL/2 = 5,7, viga "longa"). Três malhas (20/40/80 divisões): erro
  monótono decrescente, < 1 % (b2) e < 2 % (b3) na malha fina. Regra de malha:
  h ≤ π/(4λ).
- **`s-b5` — acoplamento x–y.** Com ka ≠ kt numa barra inclinada o bloco rodado
  tem termo `gxy`, e uma carga só em Y produz deslocamento também em X (1,875 e
  −3,125 mm, exatos). Uma mola `global` nunca reproduz isto; com ka = kt o bloco
  é k·I e as versões `local` e `global` coincidem a 10⁻¹².
- **Molas unilaterais.** Só os casos de análise `NonLinear` respeitam
  `mode_x`/`mode_y`; um caso `Linear` trata sempre a mola como bilateral. Cada
  modelo `s-c*` traz os dois casos (`NL` e `LIN`), pelo que a referência
  bilateral vem do mesmo ficheiro.
- **`s-c3` — sapata rígida com levantamento.** Excentricidade e = 1,5 m > L/6 =
  1,0 m: a fundação só-compressão separa-se numa parte da base. O comprimento de
  contacto discreto acerta o teórico (4,5 m) dentro de um elemento e o
  assentamento no bordo carregado converge por baixo para p_max/k = 2,667 mm
  (12/24/48 divisões: −1,1 %, −0,29 %, −0,075 %). O caso `LIN` do mesmo modelo
  traciona a fundação no bordo oposto — a diferença é o próprio objetivo do teste.

## Análise modal e espectro de resposta

Casos `m-*`, definidos em `validation/modal_cases.py` (módulo separado, mesma
mecânica `Case`/`Quantity`); teste em `tests/tests_model/test_modal_validation.py`.

Três propriedades do motor decidem o que pode ser exato e o que só pode ser
verificado por convergência:

1. **A matriz de massa é diagonal (*lumped*)**, construída a partir de `|Fy|/g`
   dos casos de carga referidos por um caso `Mass`, mais as massas nodais
   concentradas. Massa toda concentrada em nós → frequências **exatas e
   independentes da malha**; massa distribuída → O(h²), a convergir **por baixo**.
2. **Não há inércia de rotação** salvo `mtz` explícito. Os GL sem massa são
   eliminados exatamente por condensação estática (K_zz⁻¹), pelo que não geram
   modos espúrios nem perda de condicionamento. Só se K_zz for singular (GL sem
   massa a formar um mecanismo) o motor recorre a uma massa simbólica
   `eps = m_max·10⁻⁸`, com modos espúrios a ~10⁴–10⁵ × f₁.
3. **A massa vem do valor absoluto de Fy** e é aplicada igualmente a X e a Y: uma
   carga para cima dá a mesma massa que para baixo, e uma carga horizontal não
   gera massa nenhuma. Deliberado — e fixado por teste.

| Caso | Solução teórica |
|---|---|
| `m-a1` | 1 GL: f = √(3EI/mL³)/2π (flexão) e √(EA/mL)/2π (axial) |
| `m-a2` | Mola nodal em paralelo: k_eq = k_mola + 3EI/L³ |
| `m-a3` | 2 GL: eigenproblema da matriz de flexibilidade da consola |
| `m-a4` | Bloco rígido sobre Winkler: translação ω² = kL/M; rotação ω² = (k/m̄)·S/(S+Mh²/12) |
| `m-b1` | Viga s.a.: f_n = (nπ/L)²√(EI/m̄)/2π |
| `m-b2` | Consola: βL = 1,8751 / 4,6941 / 7,8548 |
| `m-b3` | Viga livre-livre: βL = 4,7300 / 7,8532 |
| `m-b4` | Barra em vibração axial: ω_n = (2n−1)π/2L·√(E/ρ) |
| `m-b5` | Viga s.a. sobre fundação: ω_n² = [EI(nπ/L)⁴ + k]/m̄ |
| `m-e1`…`m-e3` | Espectro: Sd = Sa/ω², SRSS, CQC, interpolação |

Notas:

- **Ordenação dos modos.** O motor ordena por frequência, pelo que os modos
  axiais aparecem intercalados com os de flexão (na secção padrão, o 3.º modo de
  uma viga de 6 m é axial, a 142,9 Hz). Os testes classificam cada modo pela
  componente dominante da deformada em vez de assumirem a ordem.
- **Convergência medida.** Viga s.a. com 16 elementos: −0,000 % / −0,002 % /
  −0,009 % nos três primeiros modos. Consola, mais exigente: −0,18 % / −0,62 % /
  −1,01 %. Barra axial: −0,010 % no 1.º modo. Em todos, o erro divide-se por ≈4
  a cada refinamento — confirmação direta da ordem 2.
- **Viga livre-livre.** Um modelo sem apoios nem sequer se resolve estaticamente
  (K singular), pelo que se usa o truque clássico das molas moles: as frequências
  de corpo rígido ficam três ordens de grandeza abaixo da primeira elástica, e
  verifica-se essa separação além da convergência (48,84 Hz vs 48,98 com 32
  elementos).
- **`m-a4` — o modo de rocking exige inércia de rotação.** No contínuo, um bloco
  rígido uniforme sobre fundação uniforme tem valor próprio **duplo**: translação
  e rotação partilham ω² = kL/M, porque (kL³/12)/(ML²/12) = kL/M. O modelo
  discreto não: a rotação precisa de inércia de rotação, e com massa *lumped* ela
  tem duas parcelas — as translações dos nós (Σm_i x_i², que a regra do trapézio
  só acerta a O(h²)) e o *spin* de cada segmento em torno do seu centróide
  (Σ m_i h²/12), que **não é automático**. Dando `mtz = m_i h²/12`, o modelo
  discreto tem forma fechada exata, ω² = (k/m̄)·S/(S+Mh²/12) com S = Σm_i x_i², que
  tende para kL/M quando h→0 (2/4/8 elementos: −7,4 %, −2,7 %, −0,75 %).

  **Sem `mtz`** o modo de rocking deixa de existir como tal e funde-se com a
  translação (valor próprio duplo do contínuo, ω² = kL/M). Os GL de rotação, sem
  massa, eram antes regularizados com `eps = m_max·10⁻⁸`, o que degradava o
  condicionamento de K̃ em ~10⁸ e fazia a frequência depender da rigidez do bloco
  e da *build* do LAPACK (19,62 Hz numa máquina, 19,69 Hz noutra; 19,385 Hz em vez
  de 19,346 Hz no Linux do GitHub). Passaram a ser eliminados exatamente por
  condensação estática, e o resultado é reproduzível: 19,3464073 Hz, igual à forma
  fechada a 2×10⁻⁹. O teste `test_rocking_without_rotary_inertia_is_unreliable`
  mantém-se, com a tolerância larga.
- **Massas efetivas.** Σ m_eff sobre **todos** os modos = massa nos GL livres
  (verificado a 10⁻⁶). As frações clássicas de truncatura são reproduzidas:
  consola 0,613 / 0,188 / 0,065 da massa total; viga s.a. 8/π² = 81,06 % no 1.º
  modo e **exatamente zero** nos modos pares (antissimétricos).
- **Atenção à percentagem.** `meff_%` é relativa à massa nos **GL livres**, não à
  massa total: numa viga s.a. a massa agregada aos nós apoiados está excluída, e
  por isso a percentagem varia com a malha enquanto a massa efetiva absoluta
  converge. Há um teste que fixa exatamente esta interpretação.
- **Espectro.** Com um espectro constante, a resposta espectral de um sistema de
  1 GL tem de igualar **exatamente** a análise estática com F = m·Sa — o teste
  mais apertado da cadeia Mass → Modal → Spectrum (bate a 10⁻⁹). Verifica-se
  ainda: direção XY = SRSS de X e Y; CQC ≡ SRSS para modos bem separados
  (f₂/f₁ = 5,15 com ξ = 5 %); e interpolação linear de Sa(T).

## Cortes de secção (cuts)

Caso `cut-1`, definido em `validation_cases.py` (lista `CUT_CASES`); teste em
`tests/tests_model/test_validation_x2d.py::TestCutCases`. Reaproveita o modelo
do caso 2.2 (viga s.a., L=6, q=20 kN/m) e acrescenta dois cortes — `C_MID` a
meio vão (cai sobre o nó partilhado por E1/E2) e `C_Q` a 1/4 de vão (cai
dentro de E1) — verificando `cut_result` contra o mesmo V(x), M(x) fechado do
caso 2.2. Como o sinal de um corte depende do sentido em que foi desenhado
(ver `dev/CUT_PLAN.md`), o teste compara magnitudes, à semelhança de outros
casos com ambiguidade de sinal (ex. `cn-1`).

## Ligações multiponto (constraints)

Casos `cn-*`, definidos em `validation_cases.py` (lista `CONSTRAINT_CASES`);
teste em `tests/tests_model/test_constraints_validation.py`. Entram no documento
gerado (Parte V, capítulo 8).

Uma *constraint* relaciona graus de liberdade de nós distintos por uma equação
linear — ao contrário do apoio e da mola, que atuam num GL de cada vez. O motor
impõe-nas por **penalização**: cada equação acrescenta a K um termo muito rígido.
É aproximado (erro relativo ~10⁻⁶, não à precisão da máquina) e deixa uma reação
espúria minúscula, pelo que estes casos ficam **fora** da verificação estrita de
equilíbrio (`places=6`) dos restantes e usam tolerância relativa. A ligação
mestre–escravo exata (por transformação de GL) fica para uma versão futura.

| Caso | Solução teórica |
|---|---|
| `cn-1` | Ligação rígida (braço rígido): carga excêntrica → força axial + momento no topo. M_base = P·e; θ = P·e·H/EI; u_x = P·e·H²/2EI; escravo segue o corpo rígido do mestre |
| `cn-2` | Igualdade de DOF: duas consolas com as pontas ligadas em u_y → molas em paralelo, u_y = (P₁+P₂)/2k, reação repartida R = (P₁+P₂)/2 |

Notas:

- **`cn-1` — transferência rígida completa.** A ligação rígida exercita os três
  GL (u_x, u_y, θz): a carga vertical excêntrica chega ao topo do pilar como
  força axial (encurtamento) mais momento (rotação e deslocamento horizontal).
  O teste verifica ainda a cinemática exata do escravo — u_x,s = u_x,m − Δy·θ_m,
  u_y,s = u_y,m + Δx·θ_m, θ_s = θ_m — reproduzida a 10⁻⁶.
- **`cn-2` — repartição de carga.** A igualdade u_y,B1 = u_y,B2 põe as duas
  pontas em paralelo: cargas desiguais (20 e 60 kN) produzem a mesma flecha e
  cada apoio recebe a média (40 kN), não a carga que lhe foi aplicada. É a
  diferença face ao modelo sem ligação (pontas em proporção 1:3) que o teste fixa.
