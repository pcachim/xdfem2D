# -------------------------------------------------
# Model: continuous viga (3 spans × 6 m) with uniform load
# -------------------------------------------------

import xdfem2D as xf

# Create a new Structure2D instance
struc = xf.Structure2D()

# ------------------------------------------------------------------
# 1. Define nodes for three 6-m spans (total length 18 m)
# ------------------------------------------------------------------
nodes = [
    ('N0',   0,   0),   # left support at x = 0
    ('N1',   6,   0),
    ('N2',   12,  0),
    ('N3',   18,  0)    # right support at x = 18 m
]
for name, x, y in nodes:
    struc.addnode(name=name, x=x, y=y)

# ------------------------------------------------------------------
# 2. Add beam elements (bars) connecting the nodes
# ------------------------------------------------------------------
# The API provides `addbarelement` – it creates a bar element and lets us
# specify the section directly.
beams = [
    ('B01', 'N0', 'N1'),
    ('B02', 'N1', 'N2'),
    ('B03', 'N2', 'N3')
]
for name, nodei, nodej in beams:
    struc.addbarelement(
        id=name,
        nodei=nodei,
        nodej=nodej,
        sectionname='RECT',   # we will define this later
        hingei=False,
        hingej=False
    )

# ------------------------------------------------------------------
# 3. Define material and rectangular cross-section (0.24 m × 0.24 m)
# ------------------------------------------------------------------
struc.addmaterial(
    name='C30',
    elasticmodulus=3.3e7,   # kN/m² for concrete C30
    unitweight=25           # kg/m³ → used only if self-weight is added later
)

struc.addsection(
    name='RECT',
    materialname='C30',
    b=0.24,
    h=0.24
)

# ------------------------------------------------------------------
# 4. (Optional) Assign the section to each bar – already done via addbarelement.
# If you prefer to create bars first and then assign sections, use:
# for beam in ['B01','B02','B03']:
#     struc.assignbarsection(beam, sec)
# But because `addbarelement` accepted `sectionname`, the section is already linked.

# ------------------------------------------------------------------
# 5. Apply a uniform distributed load of 10 kN/m (downward) on every bar
#    – case identifier “G”
# ------------------------------------------------------------------
loadcase = 'G'
for beam in ['B01', 'B02', 'B03']:
    # fye, fyd are the components of the distributed force in global Y (vertical)
    # Negative sign indicates downward load.
    struc.adddistributedload(
        elementid=beam,
        loadcaseid=loadcase,
        fxe=0.0,   # horizontal component at i-end
        fxd=0.0,   # horizontal component at j-end
        fye=0.0,   # vertical component at i-end (uniform)
        fyd=-10     # vertical component at j-end = -10 kN/m → uniform 10 kN/m down
    )

# ------------------------------------------------------------------
# 6. Save the model – an *.x2d file will be created alongside this script.
# ------------------------------------------------------------------
struc.save('continuaviga3vaos.x2d')