"""
Write the COMSOL parameter tables used by the Java classes (single source: comsol/params.py,
which imports motion_artifact_sim.PARAMS). Output files are committed, so the Java classes can
run without Python; rerun this script whenever PARAMS change.

  params_mech.txt   Parameters of mos2_patch_mechanics
  params_chain.txt  Parameters of mos2_sensor_chain
  vars_chain.txt    global variables of mos2_sensor_chain
  params_debye.txt  Parameters of debye_1d
Format: one "name<TAB>COMSOL expression" per line.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import params as prm  # noqa: E402


def write(name, d):
    with open(os.path.join(HERE, name), "w") as f:
        for k, v in d.items():
            f.write(f"{k}\t{v}\n")


if __name__ == "__main__":
    m = prm.mech_params()
    m.update({"t_cap": "t_PMMA", "E_cap": "E_PMMA", "nu_cap": "nu_PMMA*nu_on"})
    write("params_mech.txt", m)
    write("params_chain.txt", prm.chain_params())
    write("vars_chain.txt", prm.chain_variables())
    d = {}
    for k in ("T_room", "epsr_w", "psi0_PB"):
        v, u, _ = prm.EXTRA[k]
        d[k] = f"{v}[{u}]" if u != "1" else f"{v}"
    d.update({"c_ion": "50[mM]", "L_PB": "40[nm]", "Vth": "R_const*T_room/F_const"})
    write("params_debye.txt", d)
    print("wrote params_mech.txt, params_chain.txt, vars_chain.txt, params_debye.txt")
