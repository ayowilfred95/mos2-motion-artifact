/*
 * debye_1d.mph : 1D Poisson-Boltzmann model of a 1:1 electrolyte (Electrostatics, AC/DC module).
 * lambda_D = (integral of V dx) / psi0, exact for the linear-regime solution V = psi0 exp(-x/lambda_D).
 * Writes comsol/data/java/raw_debye.json.
 *
 * Compile:  comsolcompile comsol/java/DebyeModel.java
 * Run:      comsol batch -inputfile comsol/java/DebyeModel.class
 */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

public class DebyeModel {

  public static void main(String[] args) {
    run();
  }

  public static Model run() {
    String root = ChainModel_findRoot();
    String data = root + "/comsol/data/java";
    new File(data).mkdirs();
    LinkedHashMap<String, String> pd = readTable(root + "/comsol/java/params_debye.txt");
    long t0 = System.currentTimeMillis();

    Model model = ModelUtil.create("debye_1d");
    model.modelPath(root + "/comsol");
    for (Map.Entry<String, String> e : pd.entrySet()) model.param().set(e.getKey(), e.getValue());
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 1);
    model.component("comp1").geom("geom1").create("i1", "Interval");
    model.component("comp1").geom("geom1").feature("i1").set("coord", new String[]{"0", "L_PB"});
    model.component("comp1").geom("geom1").run();

    model.component("comp1").physics().create("es", "Electrostatics", "geom1");
    String[] names = model.component("comp1").physics("es").feature("ccn1").properties();
    if (Arrays.asList(names).contains("epsilonr_mat"))
      model.component("comp1").physics("es").feature("ccn1").set("epsilonr_mat", "userdef");
    try {
      model.component("comp1").physics("es").feature("ccn1").set("epsilonr",
          new String[]{"epsr_w", "0", "0", "0", "epsr_w", "0", "0", "0", "epsr_w"});
    } catch (Exception e) {
      model.component("comp1").physics("es").feature("ccn1").set("epsilonr", "epsr_w");
    }
    model.component("comp1").physics("es").create("scd1", "SpaceChargeDensity", 1);
    model.component("comp1").physics("es").feature("scd1").selection().all();
    model.component("comp1").physics("es").feature("scd1").set("rhoq", "-2*F_const*c_ion*sinh(V/Vth)");
    model.component("comp1").physics("es").create("pot1", "ElectricPotential", 0);
    model.component("comp1").physics("es").feature("pot1").selection().set(new int[]{1});
    model.component("comp1").physics("es").feature("pot1").set("V0", "psi0_PB");
    model.component("comp1").physics("es").create("gnd1", "Ground", 0);
    model.component("comp1").physics("es").feature("gnd1").selection().set(new int[]{2});

    model.component("comp1").cpl().create("intop1", "Integration", "geom1");
    model.component("comp1").cpl("intop1").selection().all();
    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("lam_int", "intop1(V)/psi0_PB", "Debye length from integral of V");

    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").create("edg1", "Edge");
    model.component("comp1").mesh("mesh1").feature("edg1").selection().all();
    model.component("comp1").mesh("mesh1").feature("edg1").create("dis1", "Distribution");
    model.component("comp1").mesh("mesh1").feature("edg1").feature("dis1").selection().all();
    model.component("comp1").mesh("mesh1").feature("edg1").feature("dis1").set("numelem", "4000");
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.study("std1").feature("stat").set("useparam", true);
    model.study("std1").feature("stat").setIndex("pname", "c_ion", 0);
    model.study("std1").feature("stat").setIndex("plistarr", "50 100", 0);
    model.study("std1").feature("stat").setIndex("punit", "mM", 0);
    model.study("std1").setGenPlots(false);
    model.study("std1").setGenConv(false);
    model.study("std1").run();

    String ds = null;
    List<String> sols = new ArrayList<String>();
    for (String s : model.sol().tags()) if (model.sol(s).study().equals("std1")) sols.add(s);
    for (String d : model.result().dataset().tags()) {
      try { if (sols.contains(model.result().dataset(d).getString("solution"))) { ds = d; break; } } catch (Exception e) { }
    }
    model.result().numerical().create("gev1", "EvalGlobal");
    model.result().numerical("gev1").set("data", ds);
    model.result().numerical("gev1").set("expr", new String[]{"comp1.lam_int"});
    model.result().numerical("gev1").set("unit", new String[]{"nm"});
    model.result().numerical("gev1").set("innerinput", "all");
    double[][] r = model.result().numerical("gev1").getReal();
    List<Double> lam = new ArrayList<Double>();
    for (double[] row : r) for (double x : row) lam.add(x);

    String json = "{\n \"lambda_50mM_nm\": " + lam.get(0) + ",\n \"lambda_100mM_nm\": " + lam.get(1)
        + ",\n \"debye_s\": " + (System.currentTimeMillis() - t0) / 1000.0 + "\n}\n";
    try {
      PrintWriter pw = new PrintWriter(new FileWriter(data + "/raw_debye.json"));
      pw.print(json);
      pw.close();
      model.save(root + "/comsol/debye_1d.mph");
    } catch (Exception e) { throw new RuntimeException(e); }
    System.out.println("lambda_D: 50 mM " + lam.get(0) + " nm, 100 mM " + lam.get(1) + " nm");
    return model;
  }

  static String ChainModel_findRoot() {
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
}
