/*
 * mos2_patch_mechanics.mph : 2D plane-strain laminate mechanics (COMSOL Java API).
 *
 * Builds three components (base, capped, island), runs the mesh-convergence study, calibrates the
 * Thin-Elastic-Layer stiffness k_int to eta = 0.5 / 0.8, solves the seven structures, exports strain-field
 * images and writes  comsol/data/java/raw_mech.json  and  comsol/data/java/epk.txt  (read by ChainModel).
 *
 * Compile:  comsolcompile comsol/java/MechModel.java
 * Run:      comsol batch -inputfile comsol/java/MechModel.class     (from the project folder)
 *           or COMSOL Desktop: File > Open > MechModel.class
 * The project folder is found from $MOS2_ROOT, the working directory, or ~/Downloads/mos2_motion_artifact.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

public class MechModel {

  static Model model;
  static String ROOT, DATA, FIGS;
  static double L, LF, W_ISL;
  static final double TOL = 1e-9;
  static final double[] KAPPA = {1.0, 1 / 0.025, 1 / 0.040, 1 / 0.300, 1 / 0.030};  // ref, wrist, elbow, walking, hold
  static final String[] COMPS = {"base", "capped", "island"};
  static final String[] STRUCTS = {"Baseline: 25 um PI, 5 nm Al2O3", "Polymer encapsulation (1 um PMMA)",
      "Thinner substrate (12.5 um PI)", "Ultrathin substrate (2.5 um PET)",
      "Neutral-plane buried channel + extended gate (1 um tol.)", "Rigid island (4.2 % strain ratio)",
      "Prebent channel (1e-3 strain ratio)"};
  static final double PREBENT_RATIO = 1e-3;
  static LinkedHashMap<String, String> PM = new LinkedHashMap<String, String>();
  static LinkedHashMap<String, Object> RT = new LinkedHashMap<String, Object>();

  // ================================================================= entry points
  public static void main(String[] args) {
    run();
  }

  public static Model run() {
    ROOT = findRoot();
    DATA = ROOT + "/comsol/data/java";
    FIGS = ROOT + "/comsol/figures";
    new File(DATA).mkdirs();
    new File(FIGS).mkdirs();
    PM = readTable(ROOT + "/comsol/java/params_mech.txt");
    L = num(PM.get("L_strip"));
    LF = num(PM.get("L_flake"));
    W_ISL = num(PM.get("w_isl"));
    long t0 = System.currentTimeMillis();
    build(1);
    LinkedHashMap<String, Object> raw = solveAll();
    RT.put("mech_total_s", (System.currentTimeMillis() - t0) / 1000.0);
    raw.put("runtimes", RT);
    writeText(DATA + "/raw_mech.json", toJson(raw, 0));
    log("wrote " + DATA + "/raw_mech.json");
    return model;
  }

  // ================================================================= model construction
  static String[][] rows(String c) {
    if (c.equals("base")) return new String[][]{{"sub", "0", "t_sub", "mat_sub", "2"}};
    if (c.equals("capped")) return new String[][]{{"sub", "0", "t_sub", "mat_sub", "2"},
        {"cap", "t_sub", "t_sub+t_cap", "mat_cap", "2"}};
    return new String[][]{{"pdms", "-t_PDMS", "-t_isl", "mat_pdms", "2"}, {"isl", "-t_isl", "0", "mat_pdms", "2"},
        {"sub", "0", "t_sub", "mat_sub", "2"}};
  }

  static boolean hasIsland(String c) {
    return c.equals("island");
  }

  static double[] xBreaks(boolean island) {
    TreeSet<Double> s = new TreeSet<Double>();
    double[] a = {0.0, LF / 2 - 2e-6, LF / 2, LF / 2 + 2e-6, LF / 2 + 20e-6, LF / 2 + 200e-6, 1e-3, L / 2};
    for (double x : a) { s.add(x); s.add(-x); }
    if (island) {
      double[] b = {W_ISL / 2 - 100e-6, W_ISL / 2, W_ISL / 2 + 100e-6};
      for (double x : b) { s.add(x); s.add(-x); }
    }
    double[] out = new double[s.size()];
    int i = 0;
    for (Double x : s) out[i++] = x;
    return out;
  }

  static double hTarget(double xm, int m) {
    double ax = Math.abs(xm), h;
    if (Math.abs(ax - LF / 2) <= 2e-6) h = 0.2e-6;
    else if (ax < LF / 2 + 20e-6) h = 1e-6;
    else if (ax < LF / 2 + 200e-6) h = 10e-6;
    else if (Math.abs(ax - W_ISL / 2) <= 100e-6) h = 5e-6;
    else if (ax < 1e-3) h = 50e-6;
    else h = 150e-6;
    return h / m;
  }

  static void build(int level) {
    model = ModelUtil.create("mos2_patch_mechanics");
    model.modelPath(ROOT + "/comsol");
    for (Map.Entry<String, String> e : PM.entrySet()) model.param().set(e.getKey(), e.getValue());
    for (String c : COMPS) buildComponent(c, level);
  }

  static void box(String c, String tag, int dim, double xmin, double xmax, String ymin, String ymax) {
    model.component(c).selection().create(tag, "Box");
    model.component(c).selection(tag).set("entitydim", dim);
    model.component(c).selection(tag).set("xmin", d(xmin));
    model.component(c).selection(tag).set("xmax", d(xmax));
    model.component(c).selection(tag).set("ymin", ymin);
    model.component(c).selection(tag).set("ymax", ymax);
    model.component(c).selection(tag).set("condition", "inside");
  }

  static String lo(String y) { return "(" + y + ")-" + TOL; }
  static String hi(String y) { return "(" + y + ")+" + TOL; }

  static void buildComponent(String c, int level) {
    String g = "geom_" + c;
    String[][] R = rows(c);
    double[] xs = xBreaks(hasIsland(c));
    model.component().create(c, true);
    model.component(c).geom().create(g, 2);
    model.component(c).geom(g).lengthUnit("m");
    for (String[] r : R) {
      for (int i = 0; i < xs.length - 1; i++) {
        String t = "r_" + r[0] + "_" + i;
        model.component(c).geom(g).create(t, "Rectangle");
        model.component(c).geom(g).feature(t).set("pos", new String[]{d(xs[i]), r[1]});
        model.component(c).geom(g).feature(t).set("size", new String[]{d(xs[i + 1] - xs[i]), "(" + r[2] + ")-(" + r[1] + ")"});
      }
    }
    model.component(c).geom(g).run();

    // selections
    String ybot = R[0][1], ytop = R[R.length - 1][2], dev = "t_sub";
    for (String[] r : R) {
      box(c, "sel_dom_" + r[0], 2, -L, L, lo(r[1]), hi(r[2]));
      box(c, "sel_vend_" + r[0], 1, -L / 2 - TOL, -L / 2 + TOL, lo(r[1]), hi(r[2]));
    }
    if (hasIsland(c)) box(c, "sel_dom_island", 2, -W_ISL / 2 - TOL, W_ISL / 2 + TOL, lo("-t_isl"), hi("0"));
    for (int i = 0; i < xs.length - 1; i++) box(c, "sel_h_" + i, 1, xs[i] - TOL, xs[i + 1] + TOL, lo(ybot), hi(ybot));
    box(c, "sel_dev", 1, -L, L, lo(dev), hi(dev));
    box(c, "sel_flake", 1, -LF / 2 - TOL, LF / 2 + TOL, lo(dev), hi(dev));
    box(c, "sel_out_l", 1, -L, -LF / 2 + TOL, lo(dev), hi(dev));
    box(c, "sel_out_r", 1, LF / 2 - TOL, L, lo(dev), hi(dev));
    model.component(c).selection().create("sel_devout", "Union");
    model.component(c).selection("sel_devout").set("entitydim", 1);
    model.component(c).selection("sel_devout").set("input", new String[]{"sel_out_l", "sel_out_r"});
    box(c, "sel_endL", 1, -L / 2 - TOL, -L / 2 + TOL, lo(ybot), hi(ytop));
    box(c, "sel_endR", 1, L / 2 - TOL, L / 2 + TOL, lo(ybot), hi(ytop));
    box(c, "sel_ptB", 0, -TOL, TOL, lo("0"), hi("0"));
    box(c, "sel_ptT", 0, -TOL, TOL, lo(dev), hi(dev));

    // materials (all values are parameters)
    LinkedHashSet<String> used = new LinkedHashSet<String>();
    for (String[] r : R) used.add(r[3]);
    for (String mt : used) {
      String tag = mt + "_" + c, E, nu, sel;
      if (mt.equals("mat_sub")) { E = "E_sub"; nu = "nu_sub"; sel = "sel_dom_sub"; }
      else if (mt.equals("mat_cap")) { E = "E_cap"; nu = "nu_cap"; sel = "sel_dom_cap"; }
      else {
        E = "E_PDMS"; nu = "nu_PDMS*nu_on"; sel = "sel_dom_pdms_all";
        List<String> in = new ArrayList<String>();
        for (String[] r : R) if (r[3].equals("mat_pdms")) in.add("sel_dom_" + r[0]);
        model.component(c).selection().create(sel, "Union");
        model.component(c).selection(sel).set("entitydim", 2);
        model.component(c).selection(sel).set("input", in.toArray(new String[0]));
      }
      model.component(c).material().create(tag, "Common");
      model.component(c).material(tag).selection().named(sel);
      model.component(c).material(tag).propertyGroup("def").set("youngsmodulus", new String[]{E});
      model.component(c).material(tag).propertyGroup("def").set("poissonsratio", new String[]{nu});
      model.component(c).material(tag).propertyGroup("def").set("density", new String[]{"1000[kg/m^3]"});
    }
    if (hasIsland(c)) {
      String tag = "mat_isl_" + c;       // after PDMS in the list -> overrides it on the island domain
      model.component(c).material().create(tag, "Common");
      model.component(c).material(tag).selection().named("sel_dom_island");
      model.component(c).material(tag).propertyGroup("def").set("youngsmodulus", new String[]{"E_isl"});
      model.component(c).material(tag).propertyGroup("def").set("poissonsratio", new String[]{"nu_isl*nu_on"});
      model.component(c).material(tag).propertyGroup("def").set("density", new String[]{"2500[kg/m^3]"});
    }

    // Solid Mechanics, plane strain (2D default)
    String s = "solid_" + c;
    model.component(c).physics().create(s, "SolidMechanics", g);
    try { model.component(c).physics(s).prop("Type2D").set("Type2D", "PlaneStrain"); } catch (Exception e) { }
    configureRigidConnector(c, s, "rigL", "sel_endL", "kappa*L_strip/2", true);
    configureRigidConnector(c, s, "rigR", "sel_endR", "-kappa*L_strip/2", false);

    // bonded nanometre layers: plane-strain membranes on the device boundary
    model.component(c).physics(s).create("wc_Au", "WeakContribution", 1);
    model.component(c).physics(s).feature("wc_Au").selection().named("sel_dev");
    model.component(c).physics(s).feature("wc_Au").set("weakExpression", "-Et_Au*uTx*test(uTx)");
    model.component(c).physics(s).create("wc_out", "WeakContribution", 1);
    model.component(c).physics(s).feature("wc_out").selection().named("sel_devout");
    model.component(c).physics(s).feature("wc_out").set("weakExpression", "-(Et_MoS2 + cap_on*Et_Al2O3)*uTx*test(uTx)");

    // MoS2 flake: own axial displacement um + tangential Thin Elastic Layer k_int to the substrate surface
    String w = "wbf_" + c, us = "(u-h_m*vTx)";
    model.component(c).physics().create(w, "WeakFormBoundaryPDE", g);
    model.component(c).physics(w).selection().named("sel_flake");
    model.component(c).physics(w).field("dimensionless").field("um");
    model.component(c).physics(w).field("dimensionless").component(new String[]{"um"});
    model.component(c).physics(w).feature("wfeq1").setIndex("weak",
        "-Et_flake*umTx*test(umTx) - k_int*(um-" + us + ")*test(um-" + us + ")", 0);
    model.component(c).physics(w).create("pc1", "PointwiseConstraint", 1);
    model.component(c).physics(w).feature("pc1").selection().named("sel_flake");
    model.component(c).physics(w).feature("pc1").set("constraintExpression", us + "-um");
    if (hasProp(model.component(c).physics(w).feature("pc1").properties(), "constraintForce"))
      model.component(c).physics(w).feature("pc1").set("constraintForce", "test(" + us + "-um)");

    // probes
    model.component(c).cpl().create("aveF", "Average", g);
    model.component(c).cpl("aveF").selection().geom(g, 1);
    model.component(c).cpl("aveF").selection().named("sel_flake");
    model.component(c).cpl().create("ptB", "Integration", g);
    model.component(c).cpl("ptB").selection().geom(g, 0);
    model.component(c).cpl("ptB").selection().named("sel_ptB");
    model.component(c).cpl().create("ptT", "Integration", g);
    model.component(c).cpl("ptT").selection().geom(g, 0);
    model.component(c).cpl("ptT").selection().named("sel_ptT");
    String v = "var_" + c;
    model.component(c).variable().create(v);
    model.component(c).variable(v).set("eps_ch", "aveF(umTx)", "channel-averaged MoS2 mid-plane strain");
    model.component(c).variable(v).set("eps_surf", "aveF(uTx)", "substrate surface strain under the channel");
    model.component(c).variable(v).set("eB", "ptB(" + s + ".eXX)", "axial strain, substrate bottom, x = 0");
    model.component(c).variable(v).set("eT", "ptT(" + s + ".eXX)", "axial strain, device surface, x = 0");
    model.component(c).variable(v).set("kap_fe", "(eT-eB)/(t_sub)", "local curvature at x = 0");
    model.component(c).variable(v).set("zn_fe", "-eB/(eT-eB)*t_sub", "neutral plane height above substrate bottom");
    model.component(c).variable(v).set("epk_fe", "eps_ch/kappa", "channel strain per unit curvature");

    // mapped mesh with level-dependent distributions
    String me = "mesh_" + c;
    model.component(c).mesh().create(me);
    model.component(c).mesh(me).create("map1", "Map");
    for (int i = 0; i < xs.length - 1; i++) {
      model.component(c).mesh(me).feature("map1").create("dh_" + i, "Distribution");
      model.component(c).mesh(me).feature("map1").feature("dh_" + i).selection().named("sel_h_" + i);
    }
    for (String[] r : R) {
      model.component(c).mesh(me).feature("map1").create("dv_" + r[0], "Distribution");
      model.component(c).mesh(me).feature("map1").feature("dv_" + r[0]).selection().named("sel_vend_" + r[0]);
    }
    setMeshLevel(c, level);

    // studies: spring (bonding constraint disabled) and tied (eta = 1)
    String[] kinds = {"spring", "tied"};
    for (String kind : kinds) {
      String st = "std_" + c + "_" + kind;
      model.study().create(st);
      model.study(st).create("stat", "Stationary");
      for (String o : COMPS) {
        if (o.equals(c)) continue;
        if (Arrays.asList(model.component().tags()).contains(o)) {
          model.study(st).feature("stat").activate("solid_" + o, false);
          model.study(st).feature("stat").activate("wbf_" + o, false);
        }
      }
      model.study(st).feature("stat").set("useparam", true);
      model.study(st).feature("stat").setIndex("pname", "kappa", 0);
      model.study(st).feature("stat").setIndex("plistarr", join(KAPPA), 0);
      model.study(st).feature("stat").setIndex("punit", "1/m", 0);
      if (kind.equals("spring")) {
        model.study(st).feature("stat").set("useadvanceddisable", true);
        model.study(st).feature("stat").set("disabledphysics", new String[]{w + "/pc1"});
      }
      model.study(st).setGenPlots(false);
      model.study(st).setGenConv(false);
    }
    // components built later must also be deactivated in the studies of earlier components
    for (String o : COMPS) {
      if (o.equals(c) || !Arrays.asList(model.study().tags()).contains("std_" + o + "_spring")) continue;
      for (String kind : kinds) {
        model.study("std_" + o + "_" + kind).feature("stat").activate(s, false);
        model.study("std_" + o + "_" + kind).feature("stat").activate(w, false);
      }
    }
  }

  static void configureRigidConnector(String c, String s, String tag, String sel, String rot, boolean fixX) {
    model.component(c).physics(s).create(tag, "RigidConnector", 1);
    model.component(c).physics(s).feature(tag).selection().named(sel);
    String[] names = model.component(c).physics(s).feature(tag).properties();
    String flag = first(names, new String[]{"Direction", "PrescribedDisp", "prescribedDisp"});
    String val = first(names, new String[]{"U0", "u0"});
    if (flag == null || val == null)
      throw new RuntimeException("RigidConnector translation properties not found. Available: " + Arrays.toString(names));
    boolean[] on = {fixX, true};
    String[][] opts = {{"1", "0"}, {"prescribed", "free"}, {"true", "false"}};
    for (int k = 0; k < 2; k++) {
      boolean done = false;
      for (String[] o : opts) {
        try { model.component(c).physics(s).feature(tag).setIndex(flag, on[k] ? o[0] : o[1], k); done = true; break; }
        catch (Exception e) { }
      }
      if (!done) throw new RuntimeException("could not set " + flag + " on " + tag);
      model.component(c).physics(s).feature(tag).setIndex(val, "0", k);
    }
    String rtype = first(names, new String[]{"RotationType", "rotationType"});
    if (rtype == null)
      throw new RuntimeException("RigidConnector rotation type not found. Available: " + Arrays.toString(names));
    String[] ro = {"PrescribedRotation", "Prescribed", "prescribed"};
    for (String o : ro) {
      try { model.component(c).physics(s).feature(tag).set(rtype, o); break; } catch (Exception e) { }
    }
    names = model.component(c).physics(s).feature(tag).properties();
    String ang = first(names, new String[]{"phi0", "Phi0", "phi", "thz0", "theta0"});
    if (ang == null)
      for (String n : names) if (n.toLowerCase().startsWith("phi") || n.toLowerCase().startsWith("th")) { ang = n; break; }
    if (ang == null)
      throw new RuntimeException("RigidConnector rotation angle not found. Available: " + Arrays.toString(names));
    try { model.component(c).physics(s).feature(tag).set(ang, rot); }
    catch (Exception e) { model.component(c).physics(s).feature(tag).setIndex(ang, rot, 0); }
  }

  /** Sets the mapped-mesh distributions for a refinement level and meshes. Returns the element count. */
  static int setMeshLevel(String c, int level) {
    int m = 1 << (level - 1);
    double[] xs = xBreaks(hasIsland(c));
    String me = "mesh_" + c;
    int nh = 0, nv = 0;
    for (int i = 0; i < xs.length - 1; i++) {
      int n = Math.max(1, (int) Math.round((xs[i + 1] - xs[i]) / hTarget(0.5 * (xs[i] + xs[i + 1]), m)));
      nh += n;
      model.component(c).mesh(me).feature("map1").feature("dh_" + i).set("numelem", Integer.toString(n));
    }
    for (String[] r : rows(c)) {
      int n = Math.max(1, Integer.parseInt(r[4]) * m);
      nv += n;
      model.component(c).mesh(me).feature("map1").feature("dv_" + r[0]).set("numelem", Integer.toString(n));
    }
    model.component(c).mesh(me).run();
    return nh * nv;
  }

  // ================================================================= solves
  static void setp(String k, String v) { model.param().set(k, v); }

  static void reset(String... kv) {
    for (Map.Entry<String, String> e : PM.entrySet()) model.param().set(e.getKey(), e.getValue());
    for (int i = 0; i + 1 < kv.length; i += 2) setp(kv[i], kv[i + 1]);
  }

  static int remesh(String c, int level) {
    model.component(c).geom("geom_" + c).run();
    return setMeshLevel(c, level);
  }

  static String solve(String c, String kind) {
    String st = "std_" + c + "_" + kind;
    long t0 = System.currentTimeMillis();
    model.study(st).run();
    double dt = (System.currentTimeMillis() - t0) / 1000.0;
    String key = "mech_" + c + "_" + kind;
    RT.put(key, (RT.containsKey(key) ? (Double) RT.get(key) : 0.0) + dt);
    return datasetOf(st);
  }

  static double[] ev(String c, String kind, String name) {
    return evalGlobal(model, c + "." + name, null, datasetOf("std_" + c + "_" + kind));
  }

  static int ndofs(String c, String kind) {
    for (String s : model.sol().tags()) {
      if (model.sol(s).study().equals("std_" + c + "_" + kind)) {
        try { return model.sol(s).getU().length; } catch (Exception e) { return -1; }
      }
    }
    return -1;
  }

  static double etaSeedK(double eta, double Et) {
    double a0 = 1e-3, a1 = 1e4;
    for (int i = 0; i < 200; i++) {
      double am = Math.sqrt(a0 * a1);
      if (1 - Math.tanh(am) / am - eta > 0) a1 = am; else a0 = am;
    }
    double a = Math.sqrt(a0 * a1);
    return Math.pow(2 * a / LF, 2) * Et;
  }

  static double calibTied;

  static double ratio(double logk) {
    setp("k_int", Math.pow(10, logk) + "[N/m^3]");
    solve("base", "spring");
    return ev("base", "spring", "eps_ch")[0] / calibTied;
  }

  static double[] calibrate(double eta, String capOn, double k0, List<Object> hist) {
    reset("cap_on", capOn);
    solve("base", "tied");
    calibTied = ev("base", "tied", "eps_ch")[0];
    double x0 = Math.log10(k0), x1 = x0 + 0.3;
    double f0 = ratio(x0) - eta, f1 = ratio(x1) - eta;
    hist.add(Arrays.asList(x0, f0 + eta));
    hist.add(Arrays.asList(x1, f1 + eta));
    for (int i = 0; i < 30 && Math.abs(f1) >= 1e-5; i++) {
      double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
      x0 = x1; f0 = f1;
      x1 = x2; f1 = ratio(x2) - eta;
      hist.add(Arrays.asList(x1, f1 + eta));
    }
    return new double[]{Math.pow(10, x1), f1 + eta};
  }

  static LinkedHashMap<String, Object> solveAll() {
    LinkedHashMap<String, Object> raw = new LinkedHashMap<String, Object>();
    double Eald = num(PM.get("E_MoS2")) * num(PM.get("t_MoS2")) / (1 - Math.pow(num(PM.get("nu_MoS2")), 2))
        + num(PM.get("E_Al2O3")) * num(PM.get("t_Al2O3")) / (1 - Math.pow(num(PM.get("nu_Al2O3")), 2));
    double Ebare = num(PM.get("E_MoS2")) * num(PM.get("t_MoS2")) / (1 - Math.pow(num(PM.get("nu_MoS2")), 2));
    double etaBare = num(PM.get("eta_bare")), etaAld = num(PM.get("eta_thinALD"));
    double kSeed = etaSeedK(etaAld, Eald);

    // ---- 1. mesh convergence on the baseline (both bonding models)
    log("mesh convergence");
    reset("k_int", kSeed + "[N/m^3]");
    int[] levels = {1, 2, 3, 4, 5};
    List<LinkedHashMap<String, Object>> conv = new ArrayList<LinkedHashMap<String, Object>>();
    for (int lev : levels) {
      long t0 = System.currentTimeMillis();
      int ne = remesh("base", lev);
      solve("base", "tied");
      double et = ev("base", "tied", "epk_fe")[0];
      solve("base", "spring");
      double es = ev("base", "spring", "epk_fe")[0], zn = ev("base", "spring", "zn_fe")[0],
          kf = ev("base", "spring", "kap_fe")[0];
      LinkedHashMap<String, Object> r = new LinkedHashMap<String, Object>();
      r.put("level", lev); r.put("n_elem", ne); r.put("dofs", ndofs("base", "spring"));
      r.put("epk_tied_um", et * 1e6); r.put("epk_spring_um", es * 1e6); r.put("zn_um", zn * 1e6);
      r.put("kappa_fe_over_kappa", kf); r.put("time_s", (System.currentTimeMillis() - t0) / 1000.0);
      conv.add(r);
      log("  level " + lev + ": elements " + ne + ", eps/kappa spring " + es * 1e6 + " um, tied " + et * 1e6 + " um, z_n " + zn * 1e6 + " um");
    }
    LinkedHashMap<String, Object> fin = conv.get(conv.size() - 1);
    int prod = -1;
    for (LinkedHashMap<String, Object> r : conv) {
      double ds = ((Double) r.get("epk_spring_um") - (Double) fin.get("epk_spring_um")) / (Double) fin.get("epk_spring_um") * 100;
      double dt = ((Double) r.get("epk_tied_um") - (Double) fin.get("epk_tied_um")) / (Double) fin.get("epk_tied_um") * 100;
      r.put("d_spring_pct", ds);
      r.put("d_tied_pct", dt);
      if (prod < 0 && Math.abs(ds) < 1 && Math.abs(dt) < 1) prod = (Integer) r.get("level");
    }
    prod = prod < 0 ? levels[levels.length - 1] : Math.max(3, prod);
    raw.put("convergence", conv);
    raw.put("production_level", prod);
    printConvergence(conv, prod);
    for (String c : COMPS) remesh(c, prod);

    // ---- 2. calibrate k_int
    log("calibrating k_int");
    long tc = System.currentTimeMillis();
    List<Object> hb = new ArrayList<Object>(), ha = new ArrayList<Object>();
    double[] cb = calibrate(etaBare, "0", etaSeedK(etaBare, Ebare), hb);
    double[] ca = calibrate(etaAld, "1", kSeed, ha);
    RT.put("calibration", (System.currentTimeMillis() - tc) / 1000.0);
    LinkedHashMap<String, Object> cal = new LinkedHashMap<String, Object>();
    cal.put("k_bare", cb[0]); cal.put("eta_bare", cb[1]); cal.put("k_ALD", ca[0]); cal.put("eta_ALD", ca[1]);
    cal.put("eta_polymer", "pointwise constraint (exact bonding)"); cal.put("hist_bare", hb); cal.put("hist_ALD", ha);
    raw.put("calib", cal);
    log("  k_bare " + cb[0] + " (eta " + cb[1] + "), k_ALD " + ca[0] + " (eta " + ca[1] + ")");
    model.param().set("k_bare", cb[0] + "[N/m^3]");
    model.param().set("k_ALD", ca[0] + "[N/m^3]");
    PM.put("k_bare", cb[0] + "[N/m^3]");
    PM.put("k_ALD", ca[0] + "[N/m^3]");
    PM.put("k_int", "k_ALD");

    LinkedHashMap<String, Object> epk = new LinkedHashMap<String, Object>();
    // ---- 3. baseline + substrate variants (eta = 0.8)
    log("structures");
    reset();
    remesh("base", prod);
    String ds = solve("base", "spring");
    double[] e = ev("base", "spring", "epk_fe");
    double[] ec = ev("base", "spring", "eps_ch");
    epk.put(STRUCTS[0], e[0]);
    raw.put("zn_m", ev("base", "spring", "zn_fe")[0]);
    LinkedHashMap<String, Object> em = new LinkedHashMap<String, Object>();
    em.put("wrist", ec[1]); em.put("elbow", ec[2]); em.put("walking", ec[3]); em.put("hold", ec[4]);
    raw.put("eps_motion_direct", em);
    raw.put("baseline_eps_surf_per_kappa_m", ev("base", "spring", "eps_surf")[0]);
    exportStrainImage("base", ds, 2, new double[]{-40e-6, 40e-6, -2e-6, 27e-6}, "baseline_bent");
    solve("base", "tied");
    raw.put("baseline_tied_epk_m", ev("base", "tied", "epk_fe")[0]);

    reset("t_sub", "12.5e-6[m]");
    remesh("base", prod);
    solve("base", "spring");
    epk.put(STRUCTS[2], ev("base", "spring", "epk_fe")[0]);
    reset("t_sub", "2.5e-6[m]", "E_sub", "E_PET", "nu_sub", "nu_PET*nu_on");
    remesh("base", prod);
    solve("base", "spring");
    epk.put(STRUCTS[3], ev("base", "spring", "epk_fe")[0]);

    reset("nu_on", "0");
    remesh("base", prod);
    solve("base", "tied");
    LinkedHashMap<String, Object> nu0 = new LinkedHashMap<String, Object>();
    nu0.put("epk_tied_m", ev("base", "tied", "epk_fe")[0]);
    nu0.put("zn_m", ev("base", "tied", "zn_fe")[0]);
    raw.put("nu0_check", nu0);
    reset("k_int", "k_bare", "cap_on", "0");
    solve("base", "spring");
    raw.put("bare_epk_m", ev("base", "spring", "epk_fe")[0]);

    // ---- 4. capped stacks (eta = 1)
    reset("t_cap", "t_PMMA", "E_cap", "E_PMMA", "nu_cap", "nu_PMMA*nu_on");
    remesh("capped", prod);
    solve("capped", "tied");
    epk.put(STRUCTS[1], ev("capped", "tied", "epk_fe")[0]);
    reset("t_cap", "t_buried_cap", "E_cap", "E_PI", "nu_cap", "nu_PI*nu_on");
    remesh("capped", prod);
    ds = solve("capped", "tied");
    epk.put(STRUCTS[4], ev("capped", "tied", "epk_fe")[0]);
    raw.put("buried_zn_m", ev("capped", "tied", "zn_fe")[0]);
    exportStrainImage("capped", ds, 2, new double[]{-40e-6, 40e-6, -2e-6, 50e-6}, "buried_channel");
    reset("t_cap", "25e-6[m]", "E_cap", "E_PI", "nu_cap", "nu_PI*nu_on");
    remesh("capped", prod);
    solve("capped", "tied");
    raw.put("sym_epk_m", ev("capped", "tied", "epk_fe")[0]);

    // ---- 5. rigid island (eta = 0.8)
    reset();
    remesh("island", prod);
    ds = solve("island", "spring");
    epk.put(STRUCTS[5], ev("island", "spring", "epk_fe")[0]);
    raw.put("island_local_kappa_ratio", ev("island", "spring", "kap_fe")[0]);
    raw.put("island_ratio_to_baseline", (Double) epk.get(STRUCTS[5]) / (Double) epk.get(STRUCTS[0]));
    exportStrainImage("island", ds, 2, new double[]{-2.5e-3, 2.5e-3, -110e-6, 30e-6}, "rigid_island");

    // ---- 6. prebent channel: published ratio applied to the COMSOL baseline (no 2D geometry)
    epk.put(STRUCTS[6], (Double) epk.get(STRUCTS[0]) * PREBENT_RATIO);
    raw.put("epk", epk);
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<String, Object> x : epk.entrySet()) sb.append(x.getKey()).append('\t').append(x.getValue()).append('\n');
    writeText(DATA + "/epk.txt", sb.toString());

    // restore baseline state, solve once more and save
    reset();
    for (String c : COMPS) remesh(c, prod);
    setp("t_cap", "t_buried_cap"); setp("E_cap", "E_PI"); setp("nu_cap", "nu_PI*nu_on");
    remesh("capped", prod);
    solve("base", "spring"); solve("base", "tied"); solve("capped", "tied"); solve("island", "spring");
    try { model.save(ROOT + "/comsol/mos2_patch_mechanics.mph"); } catch (Exception ex) { throw new RuntimeException(ex); }
    log("saved comsol/mos2_patch_mechanics.mph");
    return raw;
  }

  static void printConvergence(List<LinkedHashMap<String, Object>> conv, int prod) {
    System.out.println("\nMesh convergence (baseline, kappa = 1 1/m)");
    System.out.println(String.format("%5s %9s %9s %18s %8s %16s %8s %9s %7s", "level", "elements", "DOFs",
        "eps/k spring (um)", "d%", "eps/k tied (um)", "d%", "z_n (um)", "t (s)"));
    for (LinkedHashMap<String, Object> r : conv)
      System.out.println(String.format("%5d %9d %9d %18.5f %8.3f %16.5f %8.3f %9.4f %7.1f", r.get("level"), r.get("n_elem"),
          r.get("dofs"), r.get("epk_spring_um"), r.get("d_spring_pct"), r.get("epk_tied_um"), r.get("d_tied_pct"),
          r.get("zn_um"), r.get("time_s")));
    System.out.println("production level: " + prod + "\n");
  }

  static void exportStrainImage(String c, String ds, int looplevel, double[] view, String name) {
    try {
      String pg = "pg_" + name, vt = "view_" + name, et = "img_" + name;
      if (Arrays.asList(model.result().tags()).contains(pg)) model.result().remove(pg);
      if (Arrays.asList(model.component(c).view().tags()).contains(vt)) model.component(c).view().remove(vt);
      if (Arrays.asList(model.result().export().tags()).contains(et)) model.result().export().remove(et);
      model.component(c).view().create(vt, "geom_" + c);
      String[] ax = {"xmin", "xmax", "ymin", "ymax"};
      for (int i = 0; i < 4; i++) model.component(c).view(vt).axis().set(ax[i], d(view[i]));
      String[] asp = {"preserveaspect", "equal"};
      for (String p : asp) { try { model.component(c).view(vt).axis().set(p, false); } catch (Exception ex) { } }
      model.result().create(pg, "PlotGroup2D");
      model.result(pg).set("data", ds);
      model.result(pg).set("looplevel", new String[]{Integer.toString(looplevel)});
      model.result(pg).set("view", vt);
      model.result(pg).create("surf1", "Surface");
      model.result(pg).feature("surf1").set("expr", "solid_" + c + ".eXX*100");
      model.result(pg).feature("surf1").set("unit", "1");
      model.result(pg).feature("surf1").set("descr", "axial strain (%)");
      try { model.result(pg).feature("surf1").set("colortable", "RainbowLight"); } catch (Exception ex) { }
      model.result(pg).run();
      model.result().export().create(et, "Image");
      String[] names = model.result().export(et).properties();
      model.result().export(et).set(first(names, new String[]{"sourceobject", "plotgroup"}), pg);
      model.result().export(et).set(first(names, new String[]{"pngfilename", "filename"}),
          FIGS + "/comsol_strain_" + name + ".png");
      String[][] opt = {{"size", "manualprint"}, {"unit", "mm"}, {"width", "170"}, {"height", "70"},
          {"resolution", "600"}, {"imagetype", "png"}};
      for (String[] o : opt) if (hasProp(names, o[0])) model.result().export(et).set(o[0], o[1]);
      model.result().export(et).run();
      log("  image " + name);
    } catch (Exception ex) {
      log("  image export failed for " + name + ": " + ex);
    }
  }

  // ================================================================= helpers (shared pattern in all three classes)
  static String findRoot() {
    String env = System.getenv("MOS2_ROOT");
    if (env != null && new File(env, "motion_artifact_sim.py").exists()) return env;
    File dir = new File(System.getProperty("user.dir")).getAbsoluteFile();
    while (dir != null) {
      if (new File(dir, "motion_artifact_sim.py").exists()) return dir.getPath();
      dir = dir.getParentFile();
    }
    String h = System.getProperty("user.home");
    String[] cand = {h + "/Downloads/mos2_motion_artifact", h + "/mos2_motion_artifact", h + "/Desktop/mos2_motion_artifact",
        h + "/Documents/mos2_motion_artifact"};
    for (String c : cand) if (new File(c, "motion_artifact_sim.py").exists()) return c;
    throw new RuntimeException("project folder not found: set the MOS2_ROOT environment variable");
  }

  static LinkedHashMap<String, String> readTable(String path) {
    LinkedHashMap<String, String> m = new LinkedHashMap<String, String>();
    try {
      BufferedReader br = new BufferedReader(new FileReader(path));
      String line;
      while ((line = br.readLine()) != null) {
        int i = line.indexOf('\t');
        if (i > 0) m.put(line.substring(0, i), line.substring(i + 1));
      }
      br.close();
    } catch (IOException e) { throw new RuntimeException("cannot read " + path, e); }
    return m;
  }

  /** Numeric value of "1.5e-6[m]" style strings (SI units only). */
  static double num(String s) {
    int i = s.indexOf('[');
    return Double.parseDouble((i >= 0 ? s.substring(0, i) : s).trim());
  }

  static String d(double x) { return Double.toString(x); }

  static String join(double[] a) {
    StringBuilder sb = new StringBuilder();
    for (double x : a) sb.append(sb.length() > 0 ? " " : "").append(Double.toString(x));
    return sb.toString();
  }

  static boolean hasProp(String[] names, String p) {
    for (String n : names) if (n.equals(p)) return true;
    return false;
  }

  static String first(String[] names, String[] cand) {
    for (String c : cand) if (hasProp(names, c)) return c;
    return null;
  }

  static String datasetOf(String study) {
    List<String> sols = new ArrayList<String>();
    for (String s : model.sol().tags()) if (model.sol(s).study().equals(study)) sols.add(s);
    for (String dtag : model.result().dataset().tags()) {
      try { if (sols.contains(model.result().dataset(dtag).getString("solution"))) return dtag; } catch (Exception e) { }
    }
    throw new RuntimeException("no dataset for study " + study);
  }

  static double[] evalGlobal(Model mdl, String expr, String unit, String ds) {
    String tag = "gev_tmp";
    if (Arrays.asList(mdl.result().numerical().tags()).contains(tag)) mdl.result().numerical().remove(tag);
    mdl.result().numerical().create(tag, "EvalGlobal");
    mdl.result().numerical(tag).set("data", ds);
    mdl.result().numerical(tag).set("expr", new String[]{expr});
    if (unit != null) mdl.result().numerical(tag).set("unit", new String[]{unit});
    mdl.result().numerical(tag).set("innerinput", "all");
    double[][] r = mdl.result().numerical(tag).getReal();
    mdl.result().numerical().remove(tag);
    int n = 0;
    for (double[] row : r) n += row.length;
    double[] out = new double[n];
    int k = 0;
    for (double[] row : r) for (double x : row) out[k++] = x;
    return out;
  }

  static void log(String s) {
    System.out.println(new java.text.SimpleDateFormat("HH:mm:ss").format(new Date()) + " " + s);
  }

  static void writeText(String path, String text) {
    try { PrintWriter pw = new PrintWriter(new FileWriter(path)); pw.print(text); pw.close(); }
    catch (IOException e) { throw new RuntimeException(e); }
  }

  @SuppressWarnings("unchecked")
  static String toJson(Object o, int ind) {
    String pad = new String(new char[ind + 1]).replace('\0', ' ');
    if (o == null) return "null";
    if (o instanceof Map) {
      StringBuilder sb = new StringBuilder("{\n");
      int i = 0;
      Map<String, Object> m = (Map<String, Object>) o;
      for (Map.Entry<String, Object> e : m.entrySet()) {
        sb.append(pad).append(quote(e.getKey())).append(": ").append(toJson(e.getValue(), ind + 1));
        sb.append(++i < m.size() ? ",\n" : "\n");
      }
      return sb.append(new String(new char[ind]).replace('\0', ' ')).append("}").toString();
    }
    if (o instanceof List) {
      StringBuilder sb = new StringBuilder("[");
      List<Object> l = (List<Object>) o;
      for (int i = 0; i < l.size(); i++) sb.append(i > 0 ? ", " : "").append(toJson(l.get(i), ind + 1));
      return sb.append("]").toString();
    }
    if (o instanceof Double) {
      double x = (Double) o;
      if (Double.isNaN(x)) return "NaN";
      if (Double.isInfinite(x)) return x > 0 ? "Infinity" : "-Infinity";
      return Double.toString(x);
    }
    if (o instanceof Number || o instanceof Boolean) return o.toString();
    return quote(o.toString());
  }

  static String quote(String s) {
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
  }
}
