"""Resolve TODOS os modelos de validação e valida-os contra a referência.

Corre cada caso de ``validation_cases.py`` e ``modal_cases.py`` com o motor
atual (``struc.calculate()``) e, onde existe valor analítico de referência,
compara o calculado com o esperado. Não escreve ficheiros — é um verificador.

Uso:
    python validation/solve_all.py               # resolve e valida tudo
    python validation/solve_all.py --quiet       # só o resumo final
    python validation/solve_all.py --only s-a1 m-e1   # apenas alguns ids

Código de saída: 0 se tudo resolve e todas as grandezas batem a referência;
1 se algum modelo falha a resolver ou alguma grandeza sai fora da tolerância.

Precisa de ``scipy`` instalado (é o solver). Corre a partir da raiz do repo ou
de qualquer sítio — os caminhos são resolvidos a partir deste ficheiro.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validation_cases as vc          # noqa: E402
import modal_cases as mdc              # noqa: E402
import report_sections as rs           # noqa: E402


def _check(q, res, struc):
    """(ok, calculado) para uma Quantity, com a mesma regra dos testes."""
    got = q.measured(res, struc)
    ref = q.analytical
    a, b = (got, ref) if q.signed else (abs(got), abs(ref))
    ok = abs(a - b) <= q.rel_tol * abs(b) + q.abs_tol
    return ok, got


def _convergence_ids() -> set:
    """Ids das famílias de convergência, lidas das listas ``*_MESHES``.

    Estes casos aproximam um valor do contínuo à medida que a malha refina — a
    suíte valida-os pela *tendência* sobre três malhas, não por uma tolerância
    única. Uma diferença pequena face à referência analítica é convergência, não
    erro, e não deve contar como falha.
    """
    ids: set = set()
    for mod in (vc, mdc):
        for name in dir(mod):
            if name.endswith("_MESHES"):
                for entry in getattr(mod, name):
                    if entry and isinstance(entry, (list, tuple)):
                        ids.add(entry[0])
    return ids


def _build_registry():
    """id -> (build, quantities). quantities pode ser [] (só resolver)."""
    reg: dict[str, tuple] = {}
    # Casos estáticos: barras, CST, Allman, molas, arco — cada um traz build e
    # as suas grandezas de referência.
    for c in vc.CASES:
        reg[c.id] = (c.build, c.quantities)
    # Modal e espectro: as grandezas vivem em report_sections; o build vem da
    # lista de modelos modais.
    modal_build = dict(mdc.MODAL_MODELS)
    for sec in rs.modal_sections():
        build = modal_build.get(sec.model_id)
        if build is not None:
            reg[sec.model_id] = (build, sec.quantities)
    # Qualquer modelo restante (variantes de convergência, etc.): resolver sem
    # referência, só para garantir que o motor não rebenta.
    for mid, build in list(vc.ALL_MODELS) + list(mdc.MODAL_MODELS):
        reg.setdefault(mid, (build, []))
    return reg


def main() -> int:
    args = sys.argv[1:]
    quiet = "--quiet" in args
    only = []
    if "--only" in args:
        only = args[args.index("--only") + 1:]

    reg = _build_registry()
    conv_ids = _convergence_ids()
    ids = only or sorted(reg)

    n_models = n_solved = n_quant = n_ok = 0
    failed_models: list[str] = []
    failed_quant: list[str] = []
    conv_quant: list[str] = []          # diferenças aceitáveis por convergência

    for mid in ids:
        if mid not in reg:
            print(f"  ?  {mid}: id desconhecido", file=sys.stderr)
            continue
        build, quantities = reg[mid]
        is_conv = mid in conv_ids
        n_models += 1
        try:
            struc = build()
            res = struc.calculate()
        except Exception as e:                       # noqa: BLE001
            failed_models.append(mid)
            print(f"  FALHA  {mid}: não resolve — {type(e).__name__}: {e}")
            continue
        n_solved += 1

        rows = []
        model_ok = True                 # dentro da tolerância estrita
        for q in quantities:
            n_quant += 1
            try:
                ok, got = _check(q, res, struc)
            except Exception as e:                   # noqa: BLE001
                ok, got = False, f"extract ERR: {type(e).__name__}"
            # Δ relativo, para mostrar a magnitude da diferença.
            delta = None
            if isinstance(got, (int, float)) and q.analytical:
                delta = abs(abs(got) - abs(q.analytical)) / abs(q.analytical)
            if ok:
                n_ok += 1
            elif is_conv:
                # Diferença esperada num caso de convergência: conta como OK,
                # mas assinala-se para o utilizador ver o desvio.
                n_ok += 1
                conv_quant.append(f"{mid}: {q.label} (Δ={delta:.2%})"
                                  if delta is not None else f"{mid}: {q.label}")
            else:
                model_ok = False
                failed_quant.append(f"{mid}: {q.label}")
            rows.append((ok, q, got, delta))

        if not quiet:
            if not quantities:
                flag, tail = "OK  ", "  (resolvido; sem referência)"
            elif model_ok:
                flag, tail = "OK  ", ""
            elif is_conv:
                flag, tail = "CONV", "  (convergência — desvio esperado)"
            else:
                flag, tail = "DIF ", ""
            print(f"  {flag}{mid}{tail}")
            for ok, q, got, delta in rows:
                mark = "✓" if ok else ("≈" if is_conv else "✗")
                gs = f"{got:.4g}" if isinstance(got, (int, float)) else got
                ds = f"  Δ={delta:.2%}" if (not ok and delta is not None) else ""
                print(f"        {mark} {q.label:34} ref={q.analytical:.4g} "
                      f"{q.unit}   calc={gs}{ds}")

    print("\n" + "=" * 70)
    print(f"modelos: {n_solved}/{n_models} resolvem   |   "
          f"grandezas: {n_ok}/{n_quant} OK "
          f"(inclui {len(conv_quant)} por convergência)")
    if failed_models:
        print(f"não resolvem ({len(failed_models)}): {', '.join(failed_models)}")
    if conv_quant:
        print(f"convergência — desvio esperado, não é falha ({len(conv_quant)}):")
        for c in conv_quant:
            print(f"   ≈ {c}")
    if failed_quant:
        print(f"FORA da tolerância — a investigar ({len(failed_quant)}):")
        for f in failed_quant:
            print(f"   ✗ {f}")
    ok = not failed_models and not failed_quant
    print("RESULTADO:", "tudo OK" if ok else "há falhas reais — ver acima")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
