# Validação manual dos módulos de dimensionamento (RC / aço / madeira)

Cálculo independente: as expressões dos Eurocódigos são aplicadas aqui
diretamente a partir do texto da norma, sem chamar o `eurocodepy` nem o
`xdfem2d`. Serve para confirmar, por um caminho totalmente separado, os
valores obtidos por execução real em `validation/design_cases.py` (coluna
"referência" abaixo). Onde os dois caminhos batem certo há confirmação
cruzada; onde não batem, ou o caso precisa de um procedimento com mais
passos que uma fórmula fechada não cobre, isso fica assinalado — não
escondido.

Correspondência: `validation/design_cases_manual.py` implementa estas mesmas
contas em Python e confirma-as automaticamente (29/29 comparações numéricas
OK). Este ficheiro é a versão legível, com a substituição numérica passo a
passo.

## A.1 — Betão armado (EC2)

Secção retangular b = 0.30 m, h = 0.50 m, cobrimento = 0.03 m → braço
útil **d = h − cobrimento = 0.47 m**. Betão C25/30 (fck = 25 MPa), aço
A500 (fyk = 500 MPa), γc = 1.5, γs = 1.15, αcc = 1.0.

```
fcd = αcc·fck/γc = 1.0·25/1.5 = 16.667 MPa
fyd = fyk/γs = 500/1.15 = 434.783 MPa
```

### Flexão (A1, A2, A2b, A3, A8) — método K, bloco retangular (λ=0.8, η=1.0)

Momento efetivo reduzido ao eixo da armadura de tração, com N positivo =
tração:

```
M' = Med − Ned·(d − h/2)
K  = M' / (b·d²·fcd)
0.32·α² − 0.8·α + K = 0        (α = x/d, raiz menor)
z  = d·(1 − 0.4·α)
As = M'/(fyd·z) + Ned/fyd
```

| Caso | Med [kNm] | Ned [kN] | M' [kNm] | K | α | z [m] | As [m²] | As referência [m²] |
|---|---|---|---|---|---|---|---|---|
| A1 flexão simples | 150 | 0 | 150 | 0.13581 | 0.18319 | 0.43556 | **7.9208×10⁻⁴** | 7.9208×10⁻⁴ ✓ |
| A2 N+M compressão | 150 | −300 | 216 | 0.19556 | 0.27463 | 0.41837 | **4.9746×10⁻⁴** | 4.9746×10⁻⁴ ✓ |
| A2b N+M tração | 100 | +200 | 56 | 0.05070 | 0.06507 | 0.45777 | **7.4137×10⁻⁴** | 7.4137×10⁻⁴ ✓ |
| A3 hogging | −150 | 0 | −150 | (= A1, secção simétrica) | | | **7.9208×10⁻⁴** (topo) | 7.9208×10⁻⁴ ✓ |

**A8 — pilar N-M (Med=80 kNm, Ned=−800 kN):** K = 80 − (−800)·0.22 =
256 kNm → K = 256/(0.30·0.47²·16667) = **0.23178**, valor que bate certo
com a referência. Mas α ≈ 0.335 já corresponde a uma excentricidade
pequena — o próprio `xdfem2d` sinaliza este caso ("compression-controlled:
use full M-N interaction") e passa a resolver pelo diagrama de interação
N-M completo em vez do método K de secção simplesmente armada. Por isso o
As final (1.8806×10⁻⁴ m²) **não é reproduzido** por esta fórmula simples —
só o K de entrada foi confirmado.

### Esforço transverso (A4, A4b) — EC2 §6.2.2 / §9.2.2

Sem armadura longitudinal considerada (ρl = 0) → V_Rd,c cai no mínimo
v_min:

```
k     = 1 + √(200/d[mm]) ≤ 2.0            = 1 + √(200/470) = 1.6523
v_min = 0.035·k^1.5·√fck  [MPa]           = 0.035·2.1245·5 = 0.37179 MPa
VRd,c = v_min·b·d  (m→kN)                 = 0.37179·0.30·0.47·1000 = 52.41 kN

ρw,min   = 0.08·√fck/fyk                  = 0.08·5/500 = 8.0×10⁻⁴
Asw,min/s = ρw,min·b                      = 8.0×10⁻⁴·0.30 = 2.40×10⁻⁴ m²/m
```

| Caso | Ved [kN] | VRd,c [kN] | Resultado |
|---|---|---|---|
| A4 (com mínimo) | 50 | **52.41** (ref. 52.408 ✓) | Ved ≤ VRd,c → só entra o mínimo: Asw/s = **2.40×10⁻⁴ m²/m** (ref. ✓) |
| A4b (sem mínimo) | 50 | 52.41 | Ved ≤ VRd,c → Asw/s = **0** (ref. ✓) |

### Torção pura (A5) — EC2 §6.3, analogia de parede fina fechada, θ=45°

```
t_ef = (b·h)/[2(b+h)]                     = 0.15/1.6            = 0.09375 m
A_k  = (b−t_ef)(h−t_ef)                   = 0.20625·0.40625     = 0.083789 m²
u_k  = 2[(b−t_ef)+(h−t_ef)]               = 2·0.6125            = 1.225 m
ν    = 0.6(1 − fck/250)                   = 0.6·0.9             = 0.54

TRd,max = 2·ν·fcd·A_k·t_ef·sinθcosθ       (sinθcosθ = 0.5 @ 45°)
        = 2·0.54·16667·0.083789·0.09375·0.5   = 70.70 kNm

Asw,tor/s = Ted/(2·A_k·fyd)   (cotθ=1)    = 20/(2·0.083789·434783) = 2.745×10⁻⁴ m²/m
Asl,tor   = Ted·u_k/(2·A_k·fyd)           = 20·1.225/(2·0.083789·434783) = 3.363×10⁻⁴ m²
```

Todos os cinco valores intermédios batem certo com a referência (t_ef,
A_k, TRd,max = 70.697 kNm, Asw,tor/s, Asl,tor).

## A.2 — Aço EC3 §6.2, perfil IPE300 / S275

fy = 275 MPa, γM0 = 1.0. Propriedades de secção calculadas pelo próprio
motor xdfem2d: área = 5188.06 mm², Wpl,y = 602 098 mm³, Av,z = 2054.03 mm²,
Wt = 14 555 mm³.

```
Npl,Rd   = A·fy/γM0                        = 5188.06·275/1000  = 1426.72 kN
Mpl,y,Rd = Wpl,y·fy/γM0                    = 602098·275/1e6    = 165.58 kNm
Vpl,z,Rd = Av,z·fy/(√3·γM0)                = 2054.03·275/1.732/1000 = 326.12 kN
τt,Rd    = fy/√3                           = 275/1.732          = 158.77 MPa
```

| Caso | Ação | Capacidade | Utilização | Referência |
|---|---|---|---|---|
| A6.1/A6.2 N puro | N=400 kN | Npl,Rd=1426.72 kN | **0.28036** | 0.28036 ✓ |
| A6.3 M puro | My=80 kNm | Mpl,y,Rd=165.58 kNm | **0.48316** | 0.48316 ✓ |
| A6.4 V puro | Vz=150 kN | Vpl,z,Rd=326.12 kN | **0.45995** | 0.45995 ✓ |
| A6.5 T puro | T=2 kNm → τt,Ed=137.41 MPa | τt,Rd=158.77 MPa | **0.86544** | 0.86544 ✓ |

**A6.6 — encurvadura por compressão centrada (N=400 kN, coluna L=8 m):**
não reproduzida por fórmula fechada aqui — exige Ncr de Euler (com I
calculado internamente pelo xdfem2d) e a curva de encurvadura χ (§6.3.1),
um procedimento de vários passos. Fica só a nota de que o código dá
χz = 0.964 < 1.0, ou seja, a encurvadura reduz a resistência face ao caso
de secção pura (utilização 0.280 → 0.291), fisicamente coerente para uma
coluna de 8 m.

## A.3 — Madeira EC5 §6.1/§6.2, secção 0.10×0.20 m, C24

Valores de resistência característicos (EN 338, via `eurocodepy.ec5.materials.Timber`):
ft0k = 14.5 MPa, fc0k = 21.0 MPa, fmk = 24.0 MPa, fvk = 4.0 MPa. Classe de
serviço 1, duração média → kmod = 0.8 (EC5 Tabela 3.1); γM = 1.3 (madeira
maciça, EC5 Tabela 2.3).

> Nota: a primeira tentativa usou ft0k = 14.0 MPa (valor tabelado de
> memória, incorreto) e não batia certo; o valor correto do C24 é
> ft0k = 14.5 MPa — confirmado a partir dos dados reais do `eurocodepy`.

```
ft0d = kmod·ft0k/γM   = 0.8·14.5/1.3  = 8.923 MPa
fc0d = kmod·fc0k/γM   = 0.8·21.0/1.3  = 12.923 MPa
fmd  = kmod·fmk/γM    = 0.8·24.0/1.3  = 14.769 MPa
fvd  = kmod·fvk/γM    = 0.8·4.0/1.3   = 2.462 MPa

A  = 0.10·0.20                        = 0.020 m²  = 20 000 mm²
Wy = 0.10·0.20²/6                     = 6.667×10⁻⁴ m³ = 6.667×10⁵ mm³
```

| Caso | Ação | Resistência | Utilização | Referência |
|---|---|---|---|---|
| A7.1 N tração | N=25 kN | Npl,t=ft0d·A=178.46 kN | **0.14009** | 0.14009 ✓ |
| A7.2 N compressão | N=25 kN | Npl,c=fc0d·A=258.46 kN | **0.09673** | 0.09673 ✓ |
| A7.3 M puro | My=6 kNm | Mpl=fmd·Wy=9.846 kNm | **0.60938** | 0.60938 ✓ |
| A7.4 V puro | Vz=8 kN, τ=1.5V/(b·h)=0.600 MPa | fvd=2.462 MPa | **0.24375** | 0.24375 ✓ |

**A7.5 — torção pura (T=0.5 kNm):** a fórmula usual de torção retangular
(Wtor + fator de forma kshape·fvd) dá uma utilização ≈0.34 — ordem de
grandeza diferente do valor de referência (0.00223). O `eurocodepy` usa um
modelo mais elaborado de corte+torção combinados
(`check_shear_with_torsion`) que esta fórmula simples não reproduz
corretamente. **Não verificado** — assinalado para revisão em vez de
forçar um número.

**A7.6 — compressão + encurvadura (N=25 kN, comprimentos efetivos 3.0 m em
y e z):** exige o fator de instabilidade kc (§6.3.2, com λrel a partir de
E0,05 e do raio de giração) — não reproduzido por fórmula fechada aqui.
Fica só a nota de que o código dá kc,y=0.77, kc,z=0.28 (mais esbelto em z,
coerente com a secção 0.10×0.20) e utilização 0.340 > 0.097 (A7.2 sem
encurvadura), também coerente.

## Resumo

| Grupo | Casos confirmados por fórmula fechada | Casos assinalados (procedimento mais complexo) |
|---|---|---|
| A.1 Betão | A1, A2, A2b, A3, A4, A4b, A5 (7) | A8 (K confirmado; As final não) |
| A.2 Aço | A6.1–A6.5 (5) | A6.6 (encurvadura) |
| A.3 Madeira | A7.1–A7.4 (4) | A7.5 (torção — discrepância a investigar), A7.6 (encurvadura) |

16 dos 20 casos confirmados por duas fórmulas independentes (a do
`eurocodepy` e a reescrita à mão aqui); os 4 restantes ficam
explicitamente identificados, com a razão, em vez de omitidos.
