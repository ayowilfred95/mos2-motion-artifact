"""
Write the Elmer models (ElmerGrid .grd mesh definition + ElmerSolver .sif case file).

Each model is a 2D plane-strain slice of the bent strip with the real layer stack (all layers meshed,
one quadratic element through each nanometre film). Pure bending is uniform along the strip, so a
200 um half-slice with a symmetry plane at the channel centre (x = 0) gives the same state as the full
10 mm strip, while keeping element aspect ratios below 100 in the 0.65 nm MoS2 layer (longer elements
make the thin-film equations ill-conditioned). The slice is bent by a pure moment: a linear, force-free
traction on its end face. The neutral plane and the curvature are outcomes of the solve.

Usage: python elmer/make_models.py            (writes elmer/models/<name>/)
Parameter values come from motion_artifact_sim.PARAMS and comsol/params.py (Poisson ratios).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "comsol")]
from motion_artifact_sim import PARAMS as P      # noqa: E402
from params import EXTRA                          # noqa: E402

L = EXTRA["L_strip"][0]          # 10 mm
LF = EXTRA["L_flake"][0]         # 10 um channel
R_WRIST = 25e-3                  # load chosen to give roughly the wrist curvature
# body motions of the paper (motion_artifact_sim.MOTIONS): peak radius, period of one cycle, waveform
MOTIONS = {"wrist": dict(R=25e-3, T=2.0, kind="flex"), "elbow": dict(R=40e-3, T=2.0, kind="flex"),
           "walking": dict(R=300e-3, T=1 / 1.8, kind="osc")}


def motion_factor(kind, steps):
    """MATC expression of the load factor at scanning step tx(1) = 1..steps (one full cycle)."""
    tau = f"(tx(1)-1)/{steps - 1}"
    return f"0.5*(1-cos(2*pi*{tau}))" if kind == "flex" else f"sin(2*pi*{tau})"


def nu(k):
    return EXTRA[k][0]


MAT = {  # name: (E [Pa], nu, density)
    "PI": (P["E_PI"], nu("nu_PI"), 1420), "PET": (P["E_PET"], nu("nu_PET"), 1380),
    "PMMA": (P["E_PMMA"], nu("nu_PMMA"), 1180), "Au": (P["E_Au"], nu("nu_Au"), 19300),
    "MoS2": (P["E_MoS2"], nu("nu_MoS2"), 5060), "Al2O3": (P["E_Al2O3"], nu("nu_Al2O3"), 3950),
}


L_SEG = 200e-6                   # half-slice length: x = 0 (channel centre, symmetry plane) ... 200 um
NX_SEG = 3400                    # ~59 nm elements: aspect ratio < 100 even in the 0.65 nm MoS2 layer


def x_cells(nx=NX_SEG):
    """Subcell limits (m) and element counts along the slice (uniform, fine enough for the MoS2 layer)."""
    n1 = max(1, int(round(nx * LF / 2 / L_SEG)))
    return [0.0, LF / 2, L_SEG], [n1, nx - n1]


def model(name, layers, title, nx=NX_SEG, steps=None, motion="wrist"):
    """steps: None = one static solve at wrist curvature. An integer N = a stepped wrist flexion
    (Elmer "Scanning" simulation): step i of N bends the patch by 0.5*(1 - cos(2 pi (i-1)/(N-1))) of the wrist
    curvature, i.e. one flat -> bent -> flat cycle, and writes one result file per step."""
    """layers: bottom -> top list of (material, thickness [m], elements through thickness)."""
    d = os.path.join(HERE, "models", name)
    os.makedirs(d, exist_ok=True)
    xl, xn = x_cells(nx)
    y = [0.0]
    for _, t, _ in layers:
        y.append(y[-1] + t)
    nx, ny = len(xn), len(layers)
    rows = []
    for i in reversed(range(ny)):                     # ElmerGrid lists the top row first
        rows.append("  " + " ".join(str(i + 1) for _ in range(nx)))
    bnd = ["  1 -1 1 1"]                              # 1 = bottom
    bnd += [f"  2 -2 {i + 1} 1" for i in range(ny)]  # 2 = right end (all layers)
    bnd += [f"  3 -3 {ny} 1"]                         # 3 = top
    bnd += [f"  4 -4 {i + 1} 1" for i in range(ny)]  # 4 = x = 0, symmetry plane (all layers)
    grd = f"""##### {title}: ElmerGrid structured mesh (bodies bottom->top: {', '.join(m for m, _, _ in layers)}) #####
Version = 210903
Coordinate System = Cartesian 2D
Subcell Divisions in 2D = {nx} {ny}
Subcell Limits 1 = {' '.join(repr(v) for v in xl)}
Subcell Limits 2 = {' '.join(repr(v) for v in y)}
Material Structure in 2D
{chr(10).join(rows)}
End
Materials Interval = 1 {ny}
Boundary Definitions
# type out int
{chr(10).join(bnd)}
End
Numbering = Horizontal
Element Degree = 2
Element Innernodes = False
Triangles = False
Coordinate Ratios = 1
Element Divisions 1 = {' '.join(str(n) for n in xn)}
Element Divisions 2 = {' '.join(str(n) for _, _, n in layers)}
"""
    open(os.path.join(d, f"{name}.grd"), "w").write(grd)

    # pure moment: traction s*(y - ym) on the right end, the opposite on the left; zero net force
    H = y[-1]
    ym = H / 2
    # plane-strain laminate bending stiffness D about its neutral plane -> moment M = D/R_WRIST
    Ep = [MAT[m][0] / (1 - MAT[m][1] ** 2) for m, _, _ in layers]
    zc = [0.5 * (a + b) for a, b in zip(y[:-1], y[1:])]
    ts = [t for _, t, _ in layers]
    zn = sum(e * t * z for e, t, z in zip(Ep, ts, zc)) / sum(e * t for e, t in zip(Ep, ts))
    D = sum(e * (t ** 3 / 12 + t * (z - zn) ** 2) for e, t, z in zip(Ep, ts, zc))
    s = 12 * D / R_WRIST / H ** 3                # traction s*(y - ym) carries the moment s*H^3/12
    if steps:
        mo = MOTIONS[motion]
        sm = s * R_WRIST / mo["R"]                  # peak load for this motion's peak curvature
        sim_type = f"Simulation Type = Scanning\n  Timestep Intervals(1) = {steps}"
        force = ("Force 1 = Variable Coordinate 2, Time\n"
                 f'    Real MATC "{sm!r}*(tx(0)-{ym!r})*{motion_factor(mo["kind"], steps)}"')
    else:
        sim_type = "Simulation Type = Steady state"
        force = f'Force 1 = Variable Coordinate 2\n    Real MATC "{s!r}*(tx-{ym!r})"'
    bodies = "\n".join(f"""Body {i + 1}
  Name = "{m}"
  Target Bodies(1) = {i + 1}
  Equation = 1
  Material = {i + 1}
End
""" for i, (m, _, _) in enumerate(layers))
    mats = "\n".join(f"""Material {i + 1}
  Name = "{m}"
  Youngs Modulus = {MAT[m][0]!r}
  Poisson Ratio = {MAT[m][1]!r}
  Density = {MAT[m][2]}
End
""" for i, (m, _, _) in enumerate(layers))
    ych = y[2] + layers[2][1] / 2 if len(layers) > 2 else H      # MoS2 mid-plane (3rd layer in the stack)
    sif = f"""! {title}
! 2D plane strain, linear elasticity, pure bending by a force-free linear end traction.
! Generated by elmer/make_models.py - edit there, not here.
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "{name}"
  Results Directory "results"
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 2D
  {sim_type}
  Steady State Max Iterations = 1
  Output Intervals(1) = 1
End

Constants
End

{bodies}
Equation 1
  Name = "Linear elasticity (plane strain)"
  Active Solvers(3) = 1 2 3
End

Solver 1
  Equation = Linear elasticity
  Procedure = "StressSolve" "StressSolver"
  Variable = -dofs 2 Displacement
  Plane Stress = False
  Displace Mesh = False
  Calculate Strains = True
  Calculate Stresses = True
  Linear System Solver = Direct
  Linear System Direct Method = Umfpack
End

Solver 2
  Equation = SaveLine
  Procedure = "SaveData" "SaveLine"
  Filename = "lines.dat"
  Polyline Coordinates(4,2) = Real 1.0e-9 0.0  1.0e-9 {H!r}  50.0e-6 0.0  50.0e-6 {H!r}
  Polyline Divisions(2) = 2000 2000
End

Solver 3
  Equation = ResultOutput
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "{name}"
  Vtu Format = True
  Ascii Output = True
End

{mats}
! moment load: traction = s*(y - y_mid), s = {s:.6e} Pa/m, y_mid = {ym!r} m
Boundary Condition 1
  Name = "right end: moment"
  Target Boundaries(1) = 2
  {force}
End

Boundary Condition 2
  Name = "x = 0: symmetry plane (channel centre)"
  Target Boundaries(1) = 4
  Displacement 1 = 0.0
End

Boundary Condition 3
  Name = "pin at x=0, bottom (no vertical rigid motion)"
  Target Coordinates(1,2) = 0.0 0.0
  Displacement 2 = 0.0
End
"""
    open(os.path.join(d, "case.sif"), "w").write(sif)
    open(os.path.join(d, "ELMERSOLVER_STARTINFO"), "w").write("case.sif\n1\n")
    return d, dict(y=y, H=H, y_mos2=ych)


LAYERS_BASE = [("PI", 25e-6, 8), ("Au", P["t_Au"], 1), ("MoS2", P["t_MoS2"], 1), ("Al2O3", P["t_Al2O3"], 1)]

if __name__ == "__main__":
    d, info = model("baseline", LAYERS_BASE, "Baseline stack: 25 um PI / 50 nm Au / 0.65 nm MoS2 / 5 nm Al2O3")
    print("wrote", d, info)


# ---------------------------------------------------------------- rigid island (full-length strip)
def model_island(name="rigid_island", title="Rigid island: 3 mm x 20 um island (70 GPa) in 100 um PDMS under 25 um PI",
                 scale=1, t_pdms=None, steps=None, motion="wrist", load_scale=1.0):
    """
    Half strip x = 0 (channel centre, symmetry) ... 5 mm, bent by a pure end moment. Layers bottom -> top:
    PDMS carrier (-100 ... -20 um), island row (-20 ... 0 um: island for x < 1.5 mm, PDMS beyond), PI 25 um.
    The nanometre films are left out here: their in-plane stiffness (~5e3 N/m) is 0.3 % of the island
    region's (1.4e6 N/m), and they would need ~60 nm elements over the full 5 mm. The MoS2 strain is read
    from the PI surface strain plus h_MoS2 times the local curvature (Kirchhoff, exact for bonded films).
    """
    d = os.path.join(HERE, "models", name)
    os.makedirs(d, exist_ok=True)
    tp = EXTRA["t_PDMS"][0] if t_pdms is None else t_pdms
    ti, ts, w = EXTRA["t_isl"][0], 25e-6, EXTRA["w_isl"][0] / 2
    xl = [0.0, LF / 2, w - 100e-6, w, w + 100e-6, L / 2]
    xn = [int(n * scale) for n in (5, 140, 50, 50, 170)]
    yl = [-tp, -ti, 0.0, ts]
    n_pdms = max(4, int(round((tp - ti) / 8e-6)))           # ~8 um elements in the PDMS below the island
    yn = [int(n * scale) for n in (n_pdms, 6, 8)]
    # bodies: 1 PDMS (carrier), 2 island, 3 PI, 4 PDMS beside the island (same material as 1).
    # Body 4 is separate so that ElmerGrid's side codes (-2 = +x side, -4 = -x side of a body's cells) only
    # pick the outer end faces; with a single PDMS body, "-x side of PDMS" would also catch the PDMS face
    # touching the island edge at x = 1.5 mm and clamp it. Rows listed top first.
    rows = ["  " + " ".join("3" for _ in xn),
            "  " + " ".join("2" if xl[i + 1] <= w + 1e-12 else "4" for i in range(len(xn))),
            "  " + " ".join("1" for _ in xn)]
    # 1 bottom, 2 = right end of the PI film (loaded), 3 top, 4 = symmetry plane x = 0, 5 = right end of PDMS (free)
    bnd = ["  1 -1 1 1", "  2 -2 3 1", "  5 -2 1 1", "  5 -2 4 1", "  3 -3 3 1",
           "  4 -4 1 1", "  4 -4 2 1", "  4 -4 3 1"]
    grd = f"""##### {title} #####
Version = 210903
Coordinate System = Cartesian 2D
Subcell Divisions in 2D = {len(xn)} 3
Subcell Limits 1 = {' '.join(repr(v) for v in xl)}
Subcell Limits 2 = {' '.join(repr(v) for v in yl)}
Material Structure in 2D
{chr(10).join(rows)}
End
Materials Interval = 1 4
Boundary Definitions
# type out int
{chr(10).join(bnd)}
End
Numbering = Horizontal
Element Degree = 2
Element Innernodes = False
Triangles = False
Coordinate Ratios = 1
Element Divisions 1 = {' '.join(str(n) for n in xn)}
Element Divisions 2 = {' '.join(str(n) for n in yn)}
"""
    open(os.path.join(d, f"{name}.grd"), "w").write(grd)
    E_pd, nu_pd = EXTRA["E_PDMS"][0], EXTRA["nu_PDMS"][0]
    E_is, nu_is = EXTRA["E_isl"][0], EXTRA["nu_isl"][0]
    E_pi, nu_pi, _ = MAT["PI"]
    # end moment applied through the PI film only (the 2 MPa PDMS end face is left free, otherwise the
    # traction would locally shear the soft PDMS); magnitude ~ wrist curvature in the unsupported PI
    H, ym = ts, ts / 2
    s = 12 * (E_pi / (1 - nu_pi ** 2)) * ts ** 3 / 12 / R_WRIST / H ** 3 * load_scale
    if steps:
        mo = MOTIONS[motion]
        s *= R_WRIST / mo["R"]
        isl_sim = f"Simulation Type = Scanning\n  Timestep Intervals(1) = {steps}"
        isl_force = ("Force 1 = Variable Coordinate 2, Time\n"
                     f'    Real MATC "{s!r}*(tx(0)-({ym!r}))*{motion_factor(mo["kind"], steps)}"')
    else:
        isl_sim = "Simulation Type = Steady state"
        isl_force = f'Force 1 = Variable Coordinate 2\n    Real MATC "{s!r}*(tx-({ym!r}))"'
    sif = f"""! {title}
! Generated by elmer/make_models.py - edit there, not here.
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "{name}"
  Results Directory "results"
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 2D
  {isl_sim}
  Steady State Max Iterations = 1
  Output Intervals(1) = 1
End

Body 1
  Name = "PDMS"
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
End

Body 2
  Name = "rigid island"
  Target Bodies(1) = 2
  Equation = 1
  Material = 2
End

Body 3
  Name = "PI"
  Target Bodies(1) = 3
  Equation = 1
  Material = 3
End

Body 4
  Name = "PDMS beside the island"
  Target Bodies(1) = 4
  Equation = 1
  Material = 1
End

Equation 1
  Name = "Linear elasticity (plane strain)"
  Active Solvers(3) = 1 2 3
End

Solver 1
  Equation = Linear elasticity
  Procedure = "StressSolve" "StressSolver"
  Variable = -dofs 2 Displacement
  Plane Stress = False
  Displace Mesh = False
  Calculate Strains = True
  Calculate Stresses = True
  Linear System Solver = Direct
  Linear System Direct Method = Umfpack
End

Solver 2
  Equation = SaveLine
  Procedure = "SaveData" "SaveLine"
  Filename = "lines.dat"
  ! 1: through the thickness at the channel (x = 0); 2, 3: along the bottom and top of the PI film
  Polyline Coordinates(6,2) = Real 1.0e-9 {-tp!r}  1.0e-9 {ts!r}  0.0 1.0e-9  {L / 2!r} 1.0e-9  0.0 {ts - 1e-9!r}  {L / 2!r} {ts - 1e-9!r}
  Polyline Divisions(3) = 2000 4000 4000
End

Solver 3
  Equation = ResultOutput
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "{name}"
  Vtu Format = True
  Ascii Output = True
End

Material 1
  Name = "PDMS"
  Youngs Modulus = {E_pd!r}
  Poisson Ratio = {nu_pd!r}
  Density = 965
End

Material 2
  Name = "rigid island"
  Youngs Modulus = {E_is!r}
  Poisson Ratio = {nu_is!r}
  Density = 2500
End

Material 3
  Name = "PI"
  Youngs Modulus = {E_pi!r}
  Poisson Ratio = {nu_pi!r}
  Density = 1420
End

! moment load: traction = s*(y - y_mid), s = {s:.6e} Pa/m, y_mid = {ym!r} m
Boundary Condition 1
  Name = "right end: moment"
  Target Boundaries(1) = 2
  {isl_force}
End

Boundary Condition 2
  Name = "x = 0: symmetry plane (channel centre)"
  Target Boundaries(1) = 4
  Displacement 1 = 0.0
End

Boundary Condition 3
  Name = "pin at x=0, bottom (no vertical rigid motion)"
  Target Coordinates(1,2) = 0.0 {-tp!r}
  Displacement 2 = 0.0
End
"""
    open(os.path.join(d, "case.sif"), "w").write(sif)
    open(os.path.join(d, "ELMERSOLVER_STARTINFO"), "w").write("case.sif\n1\n")
    return d, dict(y_bot=-tp, t_sub=ts, L_half=L / 2)


# ---------------------------------------------------------------- the stacks of the mitigation study
T_AU, T_M, T_AL = P["t_Au"], P["t_MoS2"], P["t_Al2O3"]
FILMS = [("Au", T_AU, 1), ("MoS2", T_M, 1), ("Al2O3", T_AL, 1)]
STACKS = {
    "baseline": ([("PI", 25e-6, 8)] + FILMS, "Baseline: 25 um PI / 50 nm Au / 0.65 nm MoS2 / 5 nm Al2O3"),
    "pi_12p5": ([("PI", 12.5e-6, 8)] + FILMS, "Thinner substrate: 12.5 um PI"),
    "pet_2p5": ([("PET", 2.5e-6, 8)] + FILMS, "Ultrathin substrate: 2.5 um PET"),
    "pmma_cap": ([("PI", 25e-6, 8)] + FILMS + [("PMMA", EXTRA["t_PMMA"][0], 4)], "Polymer encapsulation: 1 um PMMA cap"),
    "buried": ([("PI", 25e-6, 8)] + FILMS + [("PI", EXTRA["t_buried_cap"][0], 8)],
               "Buried channel: 25 um PI / films / 23 um PI (~1 um off the neutral plane)"),
    "symmetric": ([("PI", 25e-6, 8)] + FILMS + [("PI", 25e-6, 8)], "Check: symmetric 25 um PI / films / 25 um PI"),
}
