"""
xdfem2D — 2D plane frame structural analysis (Python port).

Quick start
-----------
from xdfem2d import Structure2D

struc = Structure2D()
struc.add_node('N1', 0.0, 0.0)
struc.add_node('N2', 5.0, 0.0)
struc.add_material('C30', elastic_modulus=30e6, unit_weight=25.0)
struc.add_section('S1', 'C30', b=0.3, h=0.6)
struc.add_bar_element('E1', 'N1', 'N2', 'S1')
struc.add_support('PIN', ux=True, uy=True)
struc.assign_support('N1', 'PIN')
struc.add_load_case('LC1')
struc.create_uniform_load('E1', 'LC1', 10.0, 'down')
results = struc.calculate()

# Save / load structure
struc.save('bridge.json')
struc2 = Structure2D.load('bridge.json')

# Save results
from xdfem2d import save_json, save_excel
save_excel(results, 'results.xlsx')

# RC design
from xdfem2d import design_reinforcement
rc = design_reinforcement(struc, results)

# Check a model's topology, write it back out as Python, or check a script
# that builds one — without running it
from xdfem2d import model_check, model_to_python, check_script
issues = model_check(struc)
source = model_to_python(struc)
problems = check_script(source)

What is stable
--------------
``xdfem2d.STABLE`` lists the names this package undertakes to keep;
``xdfem2d.UNSTABLE`` lists the ones that are exported and may still change.
Both are ordinary lists — read them, or check a name against them.
"""
__author__ = "Paulo Cachim"


def _resolve_version() -> str:
    """Resolve the application version, primarily from pyproject.toml.

    Search order:
      1. pyproject.toml found by walking up from this file (dev tree) or sitting
         next to the bundled executable (Nuitka ships it as data).
      2. Installed package metadata (importlib.metadata).
      3. "unknown" if nothing is found.
    """
    try:
        # Aliased: an unaliased import here puts 'Path' in the package's
        # namespace, so `from xdfem2d import Path` works and someone
        # eventually depends on it. Costs one word to prevent.
        from pathlib import Path as _Path
        here = _Path(__file__).resolve()
        for base in (here.parent, *here.parents):
            pp = base / "pyproject.toml"
            if not pp.is_file():
                continue
            text = pp.read_text(encoding="utf-8")
            try:
                import tomllib
                v = tomllib.loads(text).get("project", {}).get("version")
            except ModuleNotFoundError:
                import re
                m = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']', text)
                v = m.group(1) if m else None
            if v:
                return v
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError as _NotFound
        from importlib.metadata import version as _version
        try:
            return _version("xdfem2d")
        except _NotFound:
            pass
    except Exception:
        pass
    return "unknown"


__version__ = _resolve_version()

from .structure    import Structure2D
from .results_io   import save_json, save_excel, save_structure_excel
from .structure_io import (save_structure_json, load_structure_json)
from .structure_io_checked import (load_structure_json_checked,
                                   LoadReport, LoadIssue, format_report)
from .file_io      import save_x2d, load_x2d
from .sap2000_io   import (save_s2k, load_s2k, from_s2k,
                           AmbiguousDomainError)
from .seismic      import ec8_spectrum, ec8_design_spectrum
from .postprocess  import hermite_deformed
from .model_check  import model_check
from .script_export import to_python as model_to_python
from .script_check import check as check_script, summary as check_summary
from .rc_design    import (design_reinforcement, design_concrete_sections,
                           design_beam_bars,
                           reinforcement_envelope, RCSection,
                           design_and_store, store_reinforcement,
                           crack_and_store, store_crack,
                           crack_width_sections, crack_width_envelope)
from .models import (
    Node, Material, MaterialType, Section, BarElement,
    Support, SupportAssignment,
    NodeSpring,
    ElementSpring,
    LoadCase, PointLoad, DistributedLoad, ElementPointLoad, LoadCombination,
    ConcreteMaterial,
    SupportSet, Variant,
)
from .variants import (
    solve_variants, combine_across_variants, CombTerm, VariantResult,
)
from .import_io import import_model, import_as_variant, WeldReport
from .phasing import (
    solve_phase, solve_sequence, initial_state_from_results, add_imposed_strain,
)
from .models import ConstructionPhase, ConstructionSequence, ElementInitialState
from .workflows import (
    run_variant_combination, combination_validity, import_file_as_variant,
    run_sequence, list_overlays,
)
# import_extend, make_variant_model and current_restraints are deliberately
# not re-exported. They were reachable through this module but absent from
# __all__ — usable, undocumented, and promised to nobody. Every caller already
# imports them from xdfem2d.workflows and xdfem2d.import_io, which is where
# they stay.

# ── The public surface, in two halves ────────────────────────────────────
#
# STABLE is a promise. Once this package is on PyPI, taking a name out of it
# breaks somebody else's code and costs a major version, so it holds only what
# is worth keeping still for years: building a structure, solving it, reading
# and writing files, and designing reinforcement.
#
# UNSTABLE is everything else that is useful enough to export and not settled
# enough to freeze — the variant, phasing and workflow layers, and the script
# tools written recently. Importable, documented, and liable to change.
#
# The split is written down rather than felt, because the alternative is
# discovering after the first release that every one of sixty names was a
# commitment nobody made deliberately.
STABLE = [
    # Building and solving
    'Structure2D',
    # Data models — these appear in every script and in every saved file
    'Node', 'Material', 'MaterialType', 'Section', 'BarElement',
    'Support', 'SupportAssignment',
    'NodeSpring', 'ElementSpring',
    'LoadCase', 'PointLoad', 'DistributedLoad', 'ElementPointLoad',
    'LoadCombination', 'ConcreteMaterial',
    # Files
    'save_x2d', 'load_x2d',
    'save_structure_json', 'load_structure_json',
    'load_structure_json_checked', 'LoadReport', 'LoadIssue', 'format_report',
    'save_json', 'save_excel', 'save_structure_excel',
    # Reinforced concrete design
    'design_reinforcement', 'design_concrete_sections', 'design_beam_bars',
    'reinforcement_envelope', 'RCSection',
    'design_and_store', 'store_reinforcement',
    'crack_and_store', 'store_crack',
    'crack_width_sections', 'crack_width_envelope',
    # Seismic and post-processing
    'ec8_spectrum', 'ec8_design_spectrum', 'hermite_deformed',
    # Metadata
    '__version__', '__author__',
]

UNSTABLE = [
    # Reading other people's formats
    'save_s2k', 'load_s2k', 'from_s2k', 'AmbiguousDomainError',
    'import_model', 'import_as_variant', 'WeldReport',
    # Variants, phasing, and the headless controller above them
    'SupportSet', 'Variant',
    'solve_variants', 'combine_across_variants', 'CombTerm', 'VariantResult',
    'solve_phase', 'solve_sequence', 'initial_state_from_results',
    'add_imposed_strain',
    'ConstructionPhase', 'ConstructionSequence', 'ElementInitialState',
    'run_variant_combination', 'combination_validity', 'import_file_as_variant',
    'run_sequence', 'list_overlays',
    # Checking a model, and writing or checking a script that builds one
    'model_check', 'model_to_python', 'check_script', 'check_summary',
]

__all__ = STABLE + UNSTABLE
