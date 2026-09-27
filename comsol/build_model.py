"""
Build the three COMSOL models used as the paper's method.

  mos2_patch_mechanics.mph  2D plane-strain laminate mechanics (Solid Mechanics + boundary membranes +
                            MoS2 flake on a Thin-Elastic-Layer spring), three components:
                              base    : substrate + device layers   (baseline, 12.5 um PI, 2.5 um PET, bare)
                              capped  : substrate + device + solid cap (1 um PMMA, buried PI/PI channel)
                              island  : PDMS carrier with embedded rigid island + PI + device
  mos2_sensor_chain.mph     global ODE/DAE sensor chain (motion -> strain -> electronics -> binding -> reading)
  debye_1d.mph              1D Poisson-Boltzmann electrolyte (Electrostatics, AC/DC module)

Usage:  python comsol/build_model.py [--level N] [--cores N]
Everything numeric is a COMSOL Parameter (see params.py).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import params as prm                       # noqa: E402
from comsol_util import (box, union_sel, setp, setp_first, setidx_first, find_prop,  # noqa: E402
                         PropertyError)

L = prm.EXTRA["L_strip"][0]
LF = prm.EXTRA["L_flake"][0]
W_ISL = prm.EXTRA["w_isl"][0]
TOL = 1e-9                                  # selection tolerance (m); smallest element is > 10 nm
KAPPA_LIST = [1.0, 1 / 0.025, 1 / 0.040, 1 / 0.300, 1 / 0.030]   # kappa_ref, wrist, elbow, walking, hold
DEFAULT_LEVEL = 4

# ====================================================================== mechanics: geometry spec
def mech_specs():
    """Row definitions (bottom -> top). y expressions are COMSOL parameter expressions."""
    return {
        "base": dict(
            rows=[("sub", "0", "t_sub", "mat_sub", 2)],
            dev_y="t_sub", island=None),
        "capped": dict(
            rows=[("sub", "0", "t_sub", "mat_sub", 2), ("cap", "t_sub", "t_sub+t_cap", "mat_cap", 2)],
            dev_y="t_sub", island=None),
        "island": dict(
            rows=[("pdms", "-t_PDMS", "-t_isl", "mat_pdms", 2), ("isl", "-t_isl", "0", "mat_pdms", 2),
                  ("sub", "0", "t_sub", "mat_sub", 2)],
            dev_y="t_sub", island=("isl", W_ISL)),
    }


def x_breaks(with_island):
    a = [0.0, LF / 2 - 2e-6, LF / 2, LF / 2 + 2e-6, LF / 2 + 20e-6, LF / 2 + 200e-6, 1e-3]
    if with_island:
        a += [W_ISL / 2 - 100e-6, W_ISL / 2, W_ISL / 2 + 100e-6]
    a += [L / 2]
    pos = sorted(set(a))
    return sorted(set([-x for x in pos] + pos))


def h_target(xm, m):
    """Target horizontal element size at |x| = xm for refinement multiplier m (1, 2, 4, 8, 16)."""
    ax = abs(xm)
    if abs(ax - LF / 2) <= 2e-6:
        h = 0.2e-6                              # flake ends: resolve the shear-lag boundary layer
    elif ax < LF / 2:
        h = 1e-6
    elif ax < LF / 2 + 20e-6:
        h = 1e-6
    elif ax < LF / 2 + 200e-6:
        h = 10e-6
    elif abs(ax - W_ISL / 2) <= 100e-6:
        h = 5e-6                                # island edges
    elif ax < 1e-3:
        h = 50e-6
    else:
        h = 150e-6
    return h / m


def level_to_m(level):
    return 2 ** (level - 1)


# ====================================================================== mechanics: component
def build_mech_component(model, c, spec, level):
    j = model.java
    comp = j.component().create(c, True)
    g = comp.geom().create(f"geom_{c}", 2)
    g.lengthUnit("m")
    xs = x_breaks(spec["island"] is not None)
    rows = spec["rows"]

    # ---- geometry: one rectangle per (row, x-interval); Form Union gives shared boundaries
    for ri, (rname, y0, y1, _, _) in enumerate(rows):
        for xi in range(len(xs) - 1):
            r = g.create(f"r_{rname}_{xi}", "Rectangle")
            r.set("pos", [repr(xs[xi]), y0])
            r.set("size", [repr(xs[xi + 1] - xs[xi]), f"({y1})-({y0})"])
    g.run()

    # ---- selections
    ybot, ytop = rows[0][1], rows[-1][2]
    dev = spec["dev_y"]
    for rname, y0, y1, _, _ in rows:
        box(comp, f"sel_dom_{rname}", 2, -L, L, f"({y0})-{TOL}", f"({y1})+{TOL}", label=f"domains {rname}")
        box(comp, f"sel_vend_{rname}", 1, -L / 2 - TOL, -L / 2 + TOL, f"({y0})-{TOL}", f"({y1})+{TOL}")
    if spec["island"]:
        rname, w = spec["island"]
        y0, y1 = [r for r in rows if r[0] == rname][0][1:3]
        box(comp, "sel_dom_island", 2, -w / 2 - TOL, w / 2 + TOL, f"({y0})-{TOL}", f"({y1})+{TOL}")
    for xi in range(len(xs) - 1):
        box(comp, f"sel_h_{xi}", 1, xs[xi] - TOL, xs[xi + 1] + TOL, f"({ybot})-{TOL}", f"({ybot})+{TOL}")
    box(comp, "sel_dev", 1, -L, L, f"({dev})-{TOL}", f"({dev})+{TOL}", label="device boundary (all)")
    box(comp, "sel_flake", 1, -LF / 2 - TOL, LF / 2 + TOL, f"({dev})-{TOL}", f"({dev})+{TOL}", label="MoS2 flake")
    box(comp, "sel_out_l", 1, -L, -LF / 2 + TOL, f"({dev})-{TOL}", f"({dev})+{TOL}")
    box(comp, "sel_out_r", 1, LF / 2 - TOL, L, f"({dev})-{TOL}", f"({dev})+{TOL}")
    union_sel(comp, "sel_devout", 1, ["sel_out_l", "sel_out_r"], label="device boundary outside flake")
    box(comp, "sel_endL", 1, -L / 2 - TOL, -L / 2 + TOL, f"({ybot})-{TOL}", f"({ytop})+{TOL}", label="left end")
    box(comp, "sel_endR", 1, L / 2 - TOL, L / 2 + TOL, f"({ybot})-{TOL}", f"({ytop})+{TOL}", label="right end")
    box(comp, "sel_ptB", 0, -TOL, TOL, f"-{TOL}", f"{TOL}", label="x=0, substrate bottom")
    box(comp, "sel_ptT", 0, -TOL, TOL, f"({dev})-{TOL}", f"({dev})+{TOL}", label="x=0, device surface")

    # ---- materials (all values are parameters)
    mats = {"mat_sub": ("E_sub", "nu_sub", "sel_dom_sub"),
            "mat_cap": ("E_cap", "nu_cap", "sel_dom_cap"),
            "mat_pdms": ("E_PDMS", "nu_PDMS*nu_on", None)}
    used = {r[3] for r in rows}
    for mtag in sorted(used):
        E, nu, sel = mats[mtag]
        m = comp.material().create(f"{mtag}_{c}", "Common")
        m.label(f"{mtag} ({c})")
        if mtag == "mat_pdms":
            names = [f"sel_dom_{r[0]}" for r in rows if r[3] == "mat_pdms"]
            union_sel(comp, "sel_dom_pdms_all", 2, names)
            m.selection().named("sel_dom_pdms_all")
        else:
            m.selection().named(sel)
        pg = m.propertyGroup("def")
        pg.set("youngsmodulus", [E])
        pg.set("poissonsratio", [nu])
        pg.set("density", ["1000[kg/m^3]"])
    if spec["island"]:
        m = comp.material().create(f"mat_isl_{c}", "Common")   # later in the list -> overrides PDMS
        m.label(f"rigid island ({c})")
        m.selection().named("sel_dom_island")
        pg = m.propertyGroup("def")
        pg.set("youngsmodulus", ["E_isl"])
        pg.set("poissonsratio", ["nu_isl*nu_on"])
        pg.set("density", ["2500[kg/m^3]"])

    # ---- Solid Mechanics, plane strain
    s = comp.physics().create(f"solid_{c}", "SolidMechanics", f"geom_{c}")
    try:
        s.prop("Type2D").set("Type2D", "PlaneStrain")
    except Exception:
        pass   # plane strain is the 2D default
    # pure bending: rigid connectors on the two end faces, prescribed rotation +-kappa*L/2,
    # u_y = 0 at both ends, u_x = 0 on the left end only (axial force free -> N = 0)
    for side, sel, rot, fix_x in (("L", "sel_endL", "kappa*L_strip/2", True),
                                  ("R", "sel_endR", "-kappa*L_strip/2", False)):
        rc = s.create(f"rig{side}", "RigidConnector", 1)
        rc.selection().named(sel)
        configure_rigid_connector(rc, rot, fix_x)

    # bonded nanometre layers as plane-strain membranes on the device boundary (in-plane stiffness E t/(1-nu^2))
    wc = s.create("wc_Au", "WeakContribution", 1)
    wc.label("Au 50 nm membrane (bonded)")
    wc.selection().named("sel_dev")
    setp(wc, "weakExpression", "-Et_Au*uTx*test(uTx)")
    wc = s.create("wc_out", "WeakContribution", 1)
    wc.label("MoS2 + Al2O3 membranes outside channel (bonded)")
    wc.selection().named("sel_devout")
    setp(wc, "weakExpression", "-(Et_MoS2 + cap_on*Et_Al2O3)*uTx*test(uTx)")

    # ---- MoS2 flake (channel): own axial displacement um on the boundary, membrane stiffness Et_flake,
    #      tied to the substrate surface at MoS2 height through a tangential Thin Elastic Layer k_int.
    w = comp.physics().create(f"wbf_{c}", "WeakFormBoundaryPDE", f"geom_{c}")
    w.label("MoS2 flake membrane + thin elastic layer")
    w.selection().named("sel_flake")
    w.field("dimensionless").field("um")
    w.field("dimensionless").component(["um"])
    usurf = "(u-h_m*vTx)"
    w.feature("wfeq1").setIndex(
        "weak", f"-Et_flake*umTx*test(umTx) - k_int*(um-{usurf})*test(um-{usurf})", 0)
    pc = w.create("pc1", "PointwiseConstraint", 1)
    pc.label("perfect bonding (eta = 1)")
    pc.selection().named("sel_flake")
    setp(pc, "constraintExpression", f"{usurf}-um")
    setp_first(pc, ["constraintForce"], f"test({usurf}-um)", required=False)

    # ---- probes
    ave = comp.cpl().create("aveF", "Average", f"geom_{c}")
    ave.selection().geom(f"geom_{c}", 1)
    ave.selection().named("sel_flake")
    for tag, sel in (("ptB", "sel_ptB"), ("ptT", "sel_ptT")):
        op = comp.cpl().create(tag, "Integration", f"geom_{c}")
        op.selection().geom(f"geom_{c}", 0)
        op.selection().named(sel)
    v = comp.variable().create(f"var_{c}")
    v.set("eps_ch", "aveF(umTx)", "channel-averaged MoS2 mid-plane strain")
    v.set("eps_surf", "aveF(uTx)", "substrate surface strain under the channel")
    v.set("eB", f"ptB(solid_{c}.eXX)", "axial strain, substrate bottom, x = 0")
    v.set("eT", f"ptT(solid_{c}.eXX)", "axial strain, device surface, x = 0")
    v.set("kap_fe", "(eT-eB)/(t_sub)", "local curvature in the substrate at x = 0")
    v.set("zn_fe", "-eB/(eT-eB)*t_sub", "neutral plane height above substrate bottom")
    v.set("epk_fe", "eps_ch/kappa", "channel strain per unit curvature")

    # ---- mesh (mapped quads, level-dependent distributions)
    mesh = comp.mesh().create(f"mesh_{c}")
    mp = mesh.create("map1", "Map")
    for xi in range(len(xs) - 1):
        d = mp.create(f"dh_{xi}", "Distribution")
        d.selection().named(f"sel_h_{xi}")
    for rname, *_ in rows:
        d = mp.create(f"dv_{rname}", "Distribution")
        d.selection().named(f"sel_vend_{rname}")
    set_mesh_level(model, c, spec, level)

    # ---- studies: spring (Thin Elastic Layer active, bonding constraint disabled) and tied (eta = 1)
    for kind in ("spring", "tied"):
        st = j.study().create(f"std_{c}_{kind}")
        st.label(f"{c}: {kind}")
        step = st.create("stat", "Stationary")
        existing = [str(t) for t in j.component().tags()]
        for other in mech_specs():
            if other != c and other in existing:
                step.activate(f"solid_{other}", False)
                step.activate(f"wbf_{other}", False)
        step.set("useparam", True)
        step.setIndex("pname", "kappa", 0)
        step.setIndex("plistarr", " ".join(repr(k) for k in KAPPA_LIST), 0)
        step.setIndex("punit", "1/m", 0)
        if kind == "spring":
            step.set("useadvanceddisable", True)
            step.set("disabledphysics", [f"wbf_{c}/pc1"])
        st.setGenPlots(False)
        st.setGenConv(False)
    # studies of components built earlier must not solve this component's physics
    studies = [str(t) for t in j.study().tags()]
    for other in mech_specs():
        if other != c and f"std_{other}_spring" in studies:
            for kind in ("spring", "tied"):
                j.study(f"std_{other}_{kind}").feature("stat").activate(f"solid_{c}", False)
                j.study(f"std_{other}_{kind}").feature("stat").activate(f"wbf_{c}", False)
    return comp


def configure_rigid_connector(rc, rot_expr, fix_x):
    """Prescribed rotation + translation on a Rigid Connector (property names vary between versions)."""
    names = [str(p) for p in rc.properties()]
    # prescribed translation flags and values (component 0 = x, 1 = y)
    flag = next((n for n in ("Direction", "PrescribedDisp", "prescribedDisp") if n in names), None)
    val = next((n for n in ("U0", "u0") if n in names), None)
    if flag is None or val is None:
        raise PropertyError(f"RigidConnector translation properties not found. Available: {sorted(names)}")
    for comp_i, on in ((0, fix_x), (1, True)):
        for v in ((True, False), ("prescribed", "free"), ("1", "0")):
            try:
                rc.setIndex(flag, v[0] if on else v[1], comp_i)
                break
            except Exception:
                continue
        rc.setIndex(val, "0", comp_i)
    rtype = next((n for n in ("RotationType", "rotationType") if n in names), None)
    if rtype is None:
        raise PropertyError(f"RigidConnector rotation type property not found. Available: {sorted(names)}")
    for opt in ("PrescribedRotation", "Prescribed", "prescribed"):
        try:
            rc.set(rtype, opt)
            break
        except Exception:
            continue
    names = [str(p) for p in rc.properties()]
    cand = [n for n in ("phi0", "Phi0", "phi", "thz0", "theta0") if n in names] or find_prop(rc, r"^(phi|theta|th)")
    if not cand:
        raise PropertyError(f"RigidConnector rotation angle property not found. Available: {sorted(names)}")
    try:
        rc.set(cand[0], rot_expr)
    except Exception:
        rc.setIndex(cand[0], rot_expr, 0)


def set_mesh_level(model, c, spec, level):
    """Update the mapped-mesh distributions for refinement level (1 = coarsest)."""
    j = model.java
    m = level_to_m(level)
    xs = x_breaks(spec["island"] is not None)
    mp = j.component(c).mesh(f"mesh_{c}").feature("map1")
    for xi in range(len(xs) - 1):
        a, b = xs[xi], xs[xi + 1]
        n = max(1, int(round((b - a) / h_target(0.5 * (a + b), m))))
        mp.feature(f"dh_{xi}").set("numelem", str(n))
    for rname, _, _, _, nfac in spec["rows"]:
        mp.feature(f"dv_{rname}").set("numelem", str(max(1, nfac * m)))
    j.component(c).mesh(f"mesh_{c}").run()


def add_strain_plot(model, c, study, inner_index, view, fname_tag):
    """2D surface plot of axial strain (%) for the export of strain-field images."""
    j = model.java
    pg = j.result().create(f"pg_{fname_tag}", "PlotGroup2D")
    pg.label(f"axial strain, {fname_tag}")
    pg.set("data", study)
    pg.set("looplevel", [str(inner_index)])
    surf = pg.create("surf1", "Surface")
    surf.set("expr", f"solid_{c}.eXX*100")
    surf.set("unit", "1")
    surf.set("descr", "axial strain (%)")
    try:
        surf.set("colortable", "RainbowLight")
    except Exception:
        pass
    vtag = f"view_{fname_tag}"
    vw = j.component(c).view().create(vtag, f"geom_{c}")
    ax = vw.axis()
    for k, val in zip(("xmin", "xmax", "ymin", "ymax"), view):
        ax.set(k, repr(val))
    for p in ("preserveaspect", "equal"):
        try:
            ax.set(p, False)
        except Exception:
            pass
    pg.set("view", vtag)
    return pg


def build_mechanics(client, level=DEFAULT_LEVEL):
    model = client.create("mos2_patch_mechanics")
    j = model.java
    pp = prm.mech_params()
    pp.update({"t_cap": "t_PMMA", "E_cap": "E_PMMA", "nu_cap": "nu_PMMA*nu_on"})
    for k, v in pp.items():
        j.param().set(k, v)
    for c, spec in mech_specs().items():
        build_mech_component(model, c, spec, level)
    return model


# ====================================================================== sensor chain (global ODE/DAE)
def build_chain(client):
    model = client.create("mos2_sensor_chain")
    j = model.java
    for k, v in prm.chain_params().items():
        j.param().set(k, v)
    comp = j.component().create("comp1", True)
    g = comp.geom().create("geom1", 1)        # dummy 1D geometry: the chain is purely global
    iv = g.create("i1", "Interval")
    iv.set("coord", ["0", "1"])
    g.run()
    v = comp.variable().create("var1")
    for k, e in prm.chain_variables().items():
        v.set(k, e)
    ge = comp.physics().create("ge", "GlobalEquations")
    ge.label("aptamer binding (Langmuir kinetics)")
    f = ge.feature("ge1")
    f.setIndex("name", "theta", 0, 0)
    f.setIndex("equation", "thetat - (k_on*Cb_M*(1-theta) - k_off*theta)", 0, 0)
    f.setIndex("initialValueU", "0", 0, 0)
    f.setIndex("initialValueUt", "0", 0, 0)
    f.setIndex("description", "fractional aptamer coverage", 0, 0)
    comp.mesh().create("mesh1").run()

    # stationary study with an auxiliary sweep (the swept parameter is set by run_all.py)
    st = j.study().create("std_stat")
    st.label("stationary sweeps")
    step = st.create("stat", "Stationary")
    step.set("useparam", True)
    step.setIndex("pname", "kap_s", 0)
    step.setIndex("plistarr", "0", 0)
    step.setIndex("punit", "1/m", 0)
    st.setGenPlots(False)
    st.setGenConv(False)

    # time-dependent study: 1800 s scenario, output every 0.05 s
    st = j.study().create("std_time")
    st.label("1800 s scenario")
    step = st.create("time", "Transient")
    step.set("tlist", "range(0,0.05,1799.95)")
    st.setGenPlots(False)
    st.setGenConv(False)
    st.createAutoSequences("all")
    configure_time_solver(model, "std_time", maxstep="0.05")

    # short study for the motion waveforms (fig. 3)
    st = j.study().create("std_motion")
    st.label("motion waveforms 0-4 s")
    step = st.create("time", "Transient")
    step.set("tlist", "range(0,4/1999,4)")
    st.setGenPlots(False)
    st.setGenConv(False)
    st.createAutoSequences("all")
    configure_time_solver(model, "std_motion", maxstep="4/1999")
    return model


def configure_time_solver(model, study, maxstep):
    j = model.java
    for s in j.sol().tags():
        if str(j.sol(s).study()) != study:
            continue
        for ft in j.sol(s).feature().tags():
            fe = j.sol(s).feature(ft)
            if str(fe.getType()) != "Time":
                continue
            setp_first(fe, ["rtol"], "1e-7", required=False)
            setp_first(fe, ["tstepsbdf"], "strict", required=False)
            setp_first(fe, ["maxstepconstraintbdf"], "const", required=False)
            setp_first(fe, ["maxstepbdf"], maxstep, required=False)


# ====================================================================== Debye 1D (Poisson-Boltzmann)
def build_debye(client):
    model = client.create("debye_1d")
    j = model.java
    for k in ("T_room", "epsr_w", "psi0_PB"):
        v, u, _ = prm.EXTRA[k]
        j.param().set(k, f"{v}[{u}]" if u != "1" else f"{v}")
    j.param().set("c_ion", "50[mM]")
    j.param().set("L_PB", "40[nm]")
    j.param().set("Vth", "R_const*T_room/F_const")
    comp = j.component().create("comp1", True)
    g = comp.geom().create("geom1", 1)
    iv = g.create("i1", "Interval")
    iv.set("coord", ["0", "L_PB"])
    g.run()
    es = comp.physics().create("es", "Electrostatics", "geom1")
    ccn = es.feature("ccn1")
    setp_first(ccn, ["epsilonr_mat"], "userdef", required=False)
    try:
        ccn.set("epsilonr", ["epsr_w", "0", "0", "0", "epsr_w", "0", "0", "0", "epsr_w"])
    except Exception:
        ccn.set("epsilonr", "epsr_w")
    scd = es.create("scd1", "SpaceChargeDensity", 1)
    scd.selection().all()
    setp(scd, "rhoq", "-2*F_const*c_ion*sinh(V/Vth)")      # 1:1 electrolyte, Boltzmann ions
    pot = es.create("pot1", "ElectricPotential", 0)
    pot.selection().set([1])
    setp(pot, "V0", "psi0_PB")
    gnd = es.create("gnd1", "Ground", 0)
    gnd.selection().set([2])
    # lambda_D = integral(V dx)/psi0, exact for the linear-regime solution V = psi0 exp(-x/lambda_D)
    op = comp.cpl().create("intop1", "Integration", "geom1")
    op.selection().all()
    comp.variable().create("var1").set("lam_int", "intop1(V)/psi0_PB", "Debye length from integral of V")
    mesh = comp.mesh().create("mesh1")
    ed = mesh.create("edg1", "Edge")
    ed.selection().all()
    dis = ed.create("dis1", "Distribution")
    dis.selection().all()
    dis.set("numelem", "4000")                                 # 10 pm elements
    mesh.run()
    st = j.study().create("std1")
    step = st.create("stat", "Stationary")
    step.set("useparam", True)
    step.setIndex("pname", "c_ion", 0)
    step.setIndex("plistarr", "50 100", 0)
    step.setIndex("punit", "mM", 0)
    st.setGenPlots(False)
    st.setGenConv(False)
    return model


def main():
    import mph
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--solve", action="store_true", help="also solve (run_all.py does the full sweeps)")
    a = ap.parse_args()
    client = mph.start(cores=a.cores)
    out = HERE
    m1 = build_mechanics(client, a.level)
    if a.solve:
        for c in mech_specs():
            for kind in ("spring", "tied"):
                m1.java.study(f"std_{c}_{kind}").run()
    m1.save(os.path.join(out, "mos2_patch_mechanics.mph"))
    m2 = build_chain(client)
    if a.solve:
        m2.java.study("std_stat").run()
    m2.save(os.path.join(out, "mos2_sensor_chain.mph"))
    try:
        m3 = build_debye(client)
        if a.solve:
            m3.java.study("std1").run()
        m3.save(os.path.join(out, "debye_1d.mph"))
    except Exception as e:   # optional model: AC/DC or Semiconductor module required
        print("debye_1d.mph not built:", e)
    print("saved models in", out)


if __name__ == "__main__":
    main()
