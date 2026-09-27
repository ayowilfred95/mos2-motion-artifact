/*
 * mos2_sensor_chain.mph : motion -> strain -> electronics -> binding -> reading, as global ODE/DAE equations.
 *
 * eps/kappa for each structure comes from MechModel (comsol/data/java/epk.txt). Runs the stationary sweeps
 * and the 1800 s time-dependent scenario and writes
 *   comsol/data/java/raw_chain.json, sweeps.txt, time_baseline.txt, time_thin.txt, time_island.txt, motion.txt
 * (array files: one "name<TAB>v1,v2,..." per line; converted by comsol/import_java.py).
 *
 * Compile:  comsolcompile comsol/java/ChainModel.java
 * Run:      comsol batch -inputfile comsol/java/ChainModel.class     (after MechModel)
 */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

public class ChainModel {

  static Model model;
  static String ROOT, DATA;
  static LinkedHashMap<String, String> PC, VC, EPK;
  static LinkedHashMap<String, Object> RT = new LinkedHashMap<String, Object>();
  static final String[] STRUCTS = {"Baseline: 25 um PI, 5 nm Al2O3", "Polymer encapsulation (1 um PMMA)",
      "Thinner substrate (12.5 um PI)", "Ultrathin substrate (2.5 um PET)",
      "Neutral-plane buried channel + extended gate (1 um tol.)", "Rigid island (4.2 % strain ratio)",
      "Prebent channel (1e-3 strain ratio)"};
  static final double KW = 1 / 0.025, KE = 1 / 0.040, KWALK = 1 / 0.300, KH = 1 / 0.030;

  public static void main(String[] args) {
    run();
  }

  public static Model run() {
    ROOT = MechModel_findRoot();
    DATA = ROOT + "/comsol/data/java";
    new File(DATA).mkdirs();
    PC = readTable(ROOT + "/comsol/java/params_chain.txt");
    VC = readTable(ROOT + "/comsol/java/vars_chain.txt");
    EPK = readTable(DATA + "/epk.txt");
    long t0 = System.currentTimeMillis();
    build();
    LinkedHashMap<String, Object> raw = solveAll();
    RT.put("chain_total_s", (System.currentTimeMillis() - t0) / 1000.0);
    raw.put("runtimes", RT);
    writeText(DATA + "/raw_chain.json", toJson(raw, 0));
    try { model.save(ROOT + "/comsol/mos2_sensor_chain.mph"); } catch (Exception e) { throw new RuntimeException(e); }
    log("saved comsol/mos2_sensor_chain.mph and " + DATA + "/raw_chain.json");
    return model;
  }

  // ================================================================= model
  static void build() {
    model = ModelUtil.create("mos2_sensor_chain");
    model.modelPath(ROOT + "/comsol");
    reset();
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 1);      // dummy 1D geometry: the chain is purely global
    model.component("comp1").geom("geom1").create("i1", "Interval");
    model.component("comp1").geom("geom1").feature("i1").set("coord", new String[]{"0", "1"});
    model.component("comp1").geom("geom1").run();
    model.component("comp1").variable().create("var1");
    for (Map.Entry<String, String> e : VC.entrySet()) model.component("comp1").variable("var1").set(e.getKey(), e.getValue());
    model.component("comp1").physics().create("ge", "GlobalEquations");
    model.component("comp1").physics("ge").feature("ge1").setIndex("name", "theta", 0, 0);
    model.component("comp1").physics("ge").feature("ge1").setIndex("equation", "thetat - (k_on*Cb_M*(1-theta) - k_off*theta)", 0, 0);
    model.component("comp1").physics("ge").feature("ge1").setIndex("initialValueU", "0", 0, 0);
    model.component("comp1").physics("ge").feature("ge1").setIndex("initialValueUt", "0", 0, 0);
    model.component("comp1").physics("ge").feature("ge1").setIndex("description", "fractional aptamer coverage", 0, 0);
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std_stat");
    model.study("std_stat").create("stat", "Stationary");
    model.study("std_stat").feature("stat").set("useparam", true);
    model.study("std_stat").feature("stat").setIndex("pname", "kap_s", 0);
    model.study("std_stat").feature("stat").setIndex("plistarr", "0", 0);
    model.study("std_stat").feature("stat").setIndex("punit", "1/m", 0);
    model.study("std_stat").setGenPlots(false);
    model.study("std_stat").setGenConv(false);

    transientStudy("std_time", "range(0,0.05,1799.95)", "0.05");
    transientStudy("std_motion", "range(0,4/1999,4)", "4/1999");
  }

  static void transientStudy(String tag, String tlist, String maxstep) {
    model.study().create(tag);
    model.study(tag).create("time", "Transient");
    model.study(tag).feature("time").set("tlist", tlist);
    model.study(tag).setGenPlots(false);
    model.study(tag).setGenConv(false);
    model.study(tag).createAutoSequences("all");
    for (String s : model.sol().tags()) {
      if (!model.sol(s).study().equals(tag)) continue;
      try {
        String[] names = model.sol(s).feature("t1").properties();
        if (hasProp(names, "rtol")) model.sol(s).feature("t1").set("rtol", "1e-7");
        if (hasProp(names, "tstepsbdf")) model.sol(s).feature("t1").set("tstepsbdf", "strict");
        if (hasProp(names, "maxstepconstraintbdf")) model.sol(s).feature("t1").set("maxstepconstraintbdf", "const");
        if (hasProp(names, "maxstepbdf")) model.sol(s).feature("t1").set("maxstepbdf", maxstep);
      } catch (Exception e) {
        log("time solver settings not applied for " + tag + ": " + e);
      }
    }
  }

  static void reset(String... kv) {
    for (Map.Entry<String, String> e : PC.entrySet()) model.param().set(e.getKey(), e.getValue());
    if (EPK != null && EPK.containsKey(STRUCTS[0])) model.param().set("epk", EPK.get(STRUCTS[0]) + "[m]");
    for (int i = 0; i + 1 < kv.length; i += 2) model.param().set(kv[i], kv[i + 1]);
  }

  /** Stationary auxiliary sweep; returns one array per expression. */
  static double[][] sweep(String pname, double[] vals, String unit, String[] exprs, String[] units, String... fixed) {
    reset(fixed);
    model.study("std_stat").feature("stat").setIndex("pname", pname, 0);
    model.study("std_stat").feature("stat").setIndex("plistarr", join(vals), 0);
    model.study("std_stat").feature("stat").setIndex("punit", unit, 0);
    long t0 = System.currentTimeMillis();
    model.study("std_stat").run();
    addTime("chain_stationary", t0);
    String ds = datasetOf("std_stat");
    double[][] out = new double[exprs.length][];
    for (int i = 0; i < exprs.length; i++) out[i] = evalGlobal(exprs[i], units == null ? null : units[i], ds);
    return out;
  }

  static double[][] runTransient(String study, String[] exprs, String[] units, String... fixed) {
    reset(fixed);
    long t0 = System.currentTimeMillis();
    model.study(study).run();
    addTime("chain_" + study, t0);
    String ds = datasetOf(study);
    double[][] out = new double[exprs.length][];
    for (int i = 0; i < exprs.length; i++) out[i] = evalGlobal(exprs[i], units[i], ds);
    return out;
  }

  static String[] a(String... s) { return s; }
  static double[] v(double... x) { return x; }

  static double[] linspace(double a0, double a1, int n) {
    double[] x = new double[n];
    for (int i = 0; i < n; i++) x[i] = a0 + (a1 - a0) * i / (n - 1);
    return x;
  }

  static double[] logspace(double e0, double e1, int n) {
    double[] x = linspace(e0, e1, n);
    for (int i = 0; i < n; i++) x[i] = Math.pow(10, x[i]);
    return x;
  }

  static double[] scale(double[] x, double f) {
    double[] y = new double[x.length];
    for (int i = 0; i < x.length; i++) y[i] = x[i] * f;
    return y;
  }

  static String g(double x) {   // compact label for 0.1, 0.05, 0.3 ... (matches Python f"{x:g}")
    String s = new java.math.BigDecimal(Double.toString(x)).stripTrailingZeros().toPlainString();
    return s;
  }

  // ================================================================= solves
  static LinkedHashMap<String, Object> solveAll() {
    LinkedHashMap<String, Object> R = new LinkedHashMap<String, Object>();
    LinkedHashMap<String, Object> sc = new LinkedHashMap<String, Object>();
    log("chain: scalars");
    String[] ex = a("GF_ch", "GF_c", "GF_tot", "lambda_D", "dVT_max", "dVT_max*1", "FS", "K_D_ngml", "k_on", "V_ov",
        "K_D", "1[ng/ml]/MW", "john_sum", "li_xt");
    String[] un = a("1", "1", "1", "nm", "mV", "V", "1", "ng/ml", "1/(M*s)", "V", "nM", "nM", "1", "1");
    double[][] s = sweep("kap_s", v(0), "1/m", ex, un);
    sc.put("GF_ch_0p3", s[0][0]); sc.put("GF_c", s[1][0]); sc.put("GF_tot_0p3", s[2][0]); sc.put("lambda_D_nm", s[3][0]);
    sc.put("dVT_sweat_mV", s[4][0]); sc.put("FS_pct", s[6][0] * 100); sc.put("K_D_ngml", s[7][0]); sc.put("k_on", s[8][0]);
    sc.put("john_sum", s[12][0]); sc.put("li_xt", s[13][0]);
    R.put("FS", s[6][0]); R.put("V_ov", s[9][0]); R.put("dVT_max_V", s[5][0]); R.put("K_D_nM", s[10][0]);
    R.put("ngml_to_nM", s[11][0]);
    sc.put("tau_100ngml_s", sweep("C0", v(100), "ng/ml", a("tau_on"), a("s"))[0][0]);
    double[] rp = sweep("C0", v(num(PC.get("C_low")), num(PC.get("C_high_phys"))), "ng/ml", a("R_bio0"), a("1"))[0];
    sc.put("R_phys_low_pct", rp[0] * 100); sc.put("R_phys_high_pct", rp[1] * 100);
    double[][] dt = sweep("eps_s", v(0.007), "", a("dI_strain", "GF_ch"), a("1", "1"), "r_c", "0", "V_ov", "0.145[V]");
    sc.put("datye", dt[0][0]); sc.put("GF_ch_0p145", dt[1][0]);
    sc.put("lambda_100mM_formula_nm", sweep("I_ion", v(0.1), "M", a("lambda_D"), a("nm"))[0][0]);
    R.put("scalars", sc);

    log("chain: motions");
    String[] names = {"wrist", "elbow", "walking", "hold"};
    double[] ks = {KW, KE, KWALK, KH};
    double[][] m30 = sweep("kap_s", ks, "1/m", a("eps", "dI_strain", "C_false_lin", "C_false", "sat"),
        a("1", "1", "ng/ml", "ng/ml", "1"), "C0", "30[ng/ml]");
    double[][] m50 = sweep("kap_s", ks, "1/m", a("C_false", "sat"), a("ng/ml", "1"), "C0", "50[ng/ml]");
    LinkedHashMap<String, Object> mot = new LinkedHashMap<String, Object>();
    for (int i = 0; i < names.length; i++) {
      LinkedHashMap<String, Object> m = new LinkedHashMap<String, Object>();
      m.put("eps", m30[0][i]); m.put("dI", m30[1][i]); m.put("lin_false30", m30[2][i]);
      m.put("C_false30", m30[3][i]); m.put("sat30", m30[4][i] > 0.5);
      m.put("C_false50", m50[0][i]); m.put("sat50", m50[1][i] > 0.5);
      mot.put(names[i], m);
    }
    R.put("motions", mot);
    R.put("xtalk_floor", sweep("kap_s", v(KW), "1/m", a("xt_floor"), a("1"), "C0", "50[ng/ml]")[0][0]);

    log("chain: mitigation");
    LinkedHashMap<String, Object> mit = new LinkedHashMap<String, Object>();
    for (String n : STRUCTS) {
      double[][] r = sweep("kap_s", v(KW, KE, KWALK), "1/m", a("art_FS", "eps"), a("1", "1"), "epk", EPK.get(n) + "[m]");
      LinkedHashMap<String, Object> m = new LinkedHashMap<String, Object>();
      m.put("wrist", r[0][0]); m.put("elbow", r[0][1]); m.put("walking", r[0][2]); m.put("eps_wrist", r[1][0]);
      mit.put(n, m);
    }
    R.put("mitigation", mit);

    log("chain: sweeps");
    StringBuilder sw = new StringBuilder();
    double[] Vov = linspace(0.05, 1.0, 200);
    arr(sw, "gate_V", Vov);
    double[] rcs3 = {0.3, 0.0, 0.6};
    for (double rc : rcs3)
      arr(sw, "gate_ratio_rc" + g(rc), sweep("V_ov", Vov, "V", a("art_FS"), a("1"), "kap_s", KW + "[1/m]", "r_c", d(rc))[0]);
    double[] e4 = linspace(-0.1 / 100, 0.1 / 100, 400);
    arr(sw, "fig4a_eps", e4);
    double[] V3 = {0.1, 0.3, 0.6};
    for (double V : V3) arr(sw, "fig4a_V" + g(V), sweep("eps_s", e4, "", a("dI_strain"), a("1"), "V_ov", V + "[V]")[0]);
    arr(sw, "fig4a_chonly", sweep("eps_s", e4, "", a("dI_strain"), a("1"), "r_c", "0")[0]);
    arr(sw, "fig4a_conly", sweep("eps_s", e4, "", a("dI_strain"), a("1"), "r_c", "1")[0]);
    double[] Cg = logspace(-1, 3.2, 300);
    arr(sw, "fig4b_C", Cg);
    double[] I3 = {0.001, 0.05, 0.16};
    for (double I : I3) {
      double[][] r = sweep("C0", Cg, "ng/ml", a("R_bio0", "lambda_D"), a("1", "nm"), "I_ion", I + "[M]");
      arr(sw, "fig4b_R_I" + g(I), r[0]);
      arr(sw, "fig4b_lam_I" + g(I), r[1]);
    }
    double[] C0s = linspace(5, 140, 200);
    arr(sw, "fig5a_C0", C0s);
    String[] m3 = {"walking", "elbow", "wrist"};
    double[] k3 = {KWALK, KE, KW};
    for (int i = 0; i < 3; i++) {
      double[][] r = sweep("C0", C0s, "ng/ml", a("C_false", "sat"), a("ng/ml", "1"), "kap_s", k3[i] + "[1/m]");
      double[] cf = new double[C0s.length];
      for (int j = 0; j < cf.length; j++) cf[j] = r[1][j] > 0.5 ? Double.POSITIVE_INFINITY : r[0][j];
      arr(sw, "fig5a_cf_" + m3[i], cf);
    }
    double[] rcs = linspace(0, 0.8, 100);
    arr(sw, "fig5b_rc", rcs);
    for (int i = 0; i < 3; i++)
      arr(sw, "fig5b_" + m3[i], scale(sweep("r_c", rcs, "", a("art_FS"), a("1"), "kap_s", k3[i] + "[1/m]")[0], 100));
    double[] mism = linspace(0, 0.2, 50);
    arr(sw, "fig7b_mism", mism);
    double[][] rm = sweep("mism", mism, "", a("resid_twin", "resid_twin_noX"), a("1", "1"), "kap_s", KW + "[1/m]", "C0", "50[ng/ml]");
    arr(sw, "fig7b_resid", scale(rm[0], 100));
    arr(sw, "fig7b_resid_noX", scale(rm[1], 100));
    double[] eg = linspace(-1e-3, 1e-3, 4001);
    arr(sw, "transfer_eps", eg);
    arr(sw, "transfer_dI", sweep("eps_s", eg, "", a("dI_strain"), a("1"))[0]);
    writeText(DATA + "/sweeps.txt", sw.toString());

    log("chain: 1800 s scenario");
    String[] tex = a("t", "kap", "eps", "theta", "R_bio0", "R_sens", "R_ref", "Cb");
    String[] tun = a("s", "1/m", "1", "1", "1", "1", "1", "ng/ml");
    String[] tkey = a("t", "kap", "eps", "theta", "R_bio0", "R_sens", "R_ref", "Cb_ngml");
    String[][] runs = {{"baseline", STRUCTS[0]}, {"thin", STRUCTS[3]}, {"island", STRUCTS[5]}};
    for (String[] rr : runs) {
      double[][] r = runTransient("std_time", tex, tun, "epk", EPK.get(rr[1]) + "[m]", "mot", "0", "mism", "0.05");
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < tex.length; i++) arr(sb, tkey[i], r[i]);
      writeText(DATA + "/time_" + rr[0] + ".txt", sb.toString());
      log("  " + rr[0] + ": " + r[0].length + " points");
    }
    StringBuilder mo = new StringBuilder();
    String[] mn = {"wrist", "elbow", "walking"};
    for (int i = 0; i < 3; i++) {
      double[][] r = runTransient("std_motion", a("t", "kap", "eps"), a("s", "1/m", "1"), "mot", Integer.toString(i + 1));
      if (i == 0) arr(mo, "t", r[0]);
      arr(mo, "kap_" + mn[i], r[1]);
      arr(mo, "eps_" + mn[i], r[2]);
    }
    writeText(DATA + "/motion.txt", mo.toString());
    reset();
    return R;
  }

  // ================================================================= helpers
  static void addTime(String key, long t0) {
    double dt = (System.currentTimeMillis() - t0) / 1000.0;
    RT.put(key, (RT.containsKey(key) ? (Double) RT.get(key) : 0.0) + dt);
  }

  static void arr(StringBuilder sb, String key, double[] x) {
    sb.append(key).append('\t');
    for (int i = 0; i < x.length; i++) {
      if (i > 0) sb.append(',');
      double y = x[i];
      sb.append(Double.isInfinite(y) ? (y > 0 ? "inf" : "-inf") : Double.isNaN(y) ? "nan" : Double.toString(y));
    }
    sb.append('\n');
  }

  static String MechModel_findRoot() {
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
    } catch (IOException e) { throw new RuntimeException("cannot read " + path + " (run MechModel first?)", e); }
    return m;
  }

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

  static String datasetOf(String study) {
    List<String> sols = new ArrayList<String>();
    for (String s : model.sol().tags()) if (model.sol(s).study().equals(study)) sols.add(s);
    for (String dtag : model.result().dataset().tags()) {
      try { if (sols.contains(model.result().dataset(dtag).getString("solution"))) return dtag; } catch (Exception e) { }
    }
    throw new RuntimeException("no dataset for study " + study);
  }

  static double[] evalGlobal(String expr, String unit, String ds) {
    String tag = "gev_tmp";
    if (Arrays.asList(model.result().numerical().tags()).contains(tag)) model.result().numerical().remove(tag);
    model.result().numerical().create(tag, "EvalGlobal");
    model.result().numerical(tag).set("data", ds);
    model.result().numerical(tag).set("expr", new String[]{expr});
    if (unit != null) model.result().numerical(tag).set("unit", new String[]{unit});
    model.result().numerical(tag).set("innerinput", "all");
    double[][] r = model.result().numerical(tag).getReal();
    model.result().numerical().remove(tag);
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
