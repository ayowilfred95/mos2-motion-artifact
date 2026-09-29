"""
ParaView startup script: play an Elmer bending motion (one result file per step) as an animation, with a
live readout of the MoS2 strain computed by Elmer and the cortisol the sensor would report.

  paraview --script=show_motion.py        (run inside the motion's model folder, e.g. elmer/models/wrist_motion)

Settings come from environment variables (defaults = baseline sensor, wrist flexion):
  KAPPA_PEAK (1/m), KIND (flex = flat->bent->flat, osc = bends both ways), PERIOD (s), STEPS,
  T_SUB (substrate thickness, m; the MoS2 sits T_SUB + 50.3 nm above the bottom), ETA, WARP, YSCALE, TITLE

The readout: channel strain = eta x (Elmer strain at the MoS2 mid-plane, x = 0), then the sensor equations of
motion_artifact_sim.py (gate-dependent gauge factor, Schottky contacts, cross-talk, Langmuir binding at
equilibrium, Debye screening) at the true cortisol level TRUE_NGML.
"""
import glob
import math
import os
from paraview.simple import *  # noqa: F401,F403

E = os.environ.get
WARP = float(E("WARP", "20"))                 # visual exaggeration of the bend
YSCALE = float(E("YSCALE", "1"))              # stretch of the thickness direction (display only)
TRUE_NGML = 30.0                              # true sweat cortisol (ng/mL)
ETA = float(E("ETA", "0.8"))                  # vdW strain transfer (0.8 Al2O3-capped MoS2)
T_SUB = float(E("T_SUB", "25e-6"))
Y_MOS2 = T_SUB + 50e-9 + 0.65e-9 / 2
KAPPA_PEAK, KIND = float(E("KAPPA_PEAK", "40")), E("KIND", "flex")
N = int(E("STEPS", "21"))
DT = float(E("PERIOD", "2.0")) / (N - 1)
TITLE = E("TITLE", "Elmer: baseline sensor during one wrist flexion (flat -> R = 25 mm -> flat, 21 solved steps)")

# sensor model constants (motion_artifact_sim.py, V_ov = 0.3 V)
V = 0.30
GF = 15 + 185 * math.exp(-(V - 0.05) / 0.12)
DVT = 0.100 * math.exp(-2 / (0.304 / math.sqrt(0.05)))
NG = 1e-6 / 362.46 * 1e9
C0 = TRUE_NGML * NG
RTRUE = DVT * (C0 / (C0 + 50.0)) / V

files = sorted(glob.glob(os.path.join("results", "*_t*.vtu")))
src = XMLUnstructuredGridReader(registrationName="Elmer steps", FileName=files)
calc = Calculator(registrationName="axial strain (%)", Input=src)
calc.ResultArrayName = "axial strain (%)"
calc.Function = "strain_xx*100"
warp = WarpByVector(registrationName=f"bent shape (x{WARP:g})", Input=calc)
warp.Vectors = ["POINTS", "displacement"]
warp.ScaleFactor = WARP
line = Contour(registrationName="neutral plane (strain = 0)", Input=warp)
line.ContourBy = ["POINTS", "axial strain (%)"]
line.Isosurfaces = [0.0]

# The 0.65 nm MoS2 cells are thinner than ParaView's point-location tolerance, so the strain is sampled at
# two points inside the substrate (y = 0.8 T_SUB and T_SUB, x = 0) and extended linearly to the MoS2 mid-plane: in bending
# the axial strain is linear through the thickness, so this is exact (difference to the direct Elmer
# SaveLine value < 0.1 %).
probe = PlotOverLine(registrationName="MoS2 channel probe", Input=src)
probe.Point1 = [1e-9, 0.8 * T_SUB, 0.0]
probe.Point2 = [1e-9, T_SUB, 0.0]
probe.Resolution = 1

# readout text, re-evaluated at every animation step (numbers inserted directly into the expression)
M = "__import__('math')"
wave = f"0.5*(1-{M}.cos(2*{M}.pi*t/{DT * (N - 1)!r}))" if KIND == "flex" else f"{M}.sin(2*{M}.pi*t/{DT * (N - 1)!r})"
K = f"({KAPPA_PEAK!r}*{wave})"                                          # applied bend curvature at time t
CAPP = f"(50.0*th/(1-th)/{NG!r})"                                      # apparent cortisol, ng/mL
expr = (
    f"(lambda t, eb: (lambda e: (lambda dI: (lambda th: "
    f"'time  %.2f s    bend curvature  %.1f 1/m  (radius %s)\\n"
    f"MoS2 strain (Elmer)  %.4f %%    current change from strain  %+.2f %%\\n"
    f"true cortisol  %.0f ng/mL    sensor reads  %s' % ("
    f"t, {K}, (('%.0f mm' % (1000/abs({K})) + ('' if {K} > 0 else ', bent the other way')) if abs({K}) > 0.05 else 'flat'), e*100, dI*100, {TRUE_NGML!r}, "
    f"(('%.1f ng/mL  (error %+.1f)' % ({CAPP}, {CAPP} - {TRUE_NGML!r})) if th < 1 else 'OVER RANGE (saturated)')))"
    f"(({RTRUE!r}*(1+85*e) + dI)*{V!r}/{DVT!r}))"
    f"(1/(0.7*__import__('math').exp(-{GF!r}*e) + 0.3*__import__('math').exp(-5.5*e/0.025852)) - 1))"
    f"({ETA!r}*eb))({DT!r}*t_value, float(strain_xx[1]) + (float(strain_xx[1]) - float(strain_xx[0]))*{(Y_MOS2 - T_SUB) / (0.2 * T_SUB)!r})"
)
ann = PythonAnnotation(registrationName="readout", Input=probe)
ann.ArrayAssociation = "Point Data"
ann.Expression = expr

view = GetActiveViewOrCreate("RenderView")
disp = Show(warp, view)
ColorBy(disp, ("POINTS", "axial strain (%)"))
lut = GetColorTransferFunction("axialstrain")
lut.ApplyPreset("Cool to Warm", True)
lut.RescaleTransferFunction(-0.06, 0.06)
lut.AutomaticRescaleRangeMode = "Never"
GetOpacityTransferFunction("axialstrain").RescaleTransferFunction(-0.06, 0.06)
ldisp = Show(line, view)
ldisp.ColorArrayName = ["POINTS", ""]           # solid colour (the line is empty in flat steps)
ldisp.AmbientColor = ldisp.DiffuseColor = [0.0, 0.0, 0.0]
ldisp.LineWidth = 3.0
bar = GetScalarBar(lut, view)
bar.Title = "axial strain (%)"
bar.ComponentTitle = ""
bar.Visibility = 1
adisp = Show(ann, view)
adisp.WindowLocation = "Upper Left Corner"
adisp.FontSize = 18
adisp.Color = [1.0, 1.0, 1.0]
title = Text(registrationName="title")
title.Text = (TITLE + "\n"
              f"red = stretched, blue = compressed, black line = neutral plane; bend exaggerated x{WARP:g}" + (f", thickness stretched x{YSCALE:g}" if YSCALE != 1 else "") + ". "
              "Press the green Play button.")
tdisp = Show(title, view)
tdisp.WindowLocation = "Lower Center"
tdisp.FontSize = 13
for dsp in (disp, ldisp):
    dsp.Scale = [1.0, YSCALE, 1.0]
Hide(src, view)
Hide(calc, view)
Hide(probe, view)
view.InteractionMode = "2D"
view.OrientationAxesVisibility = 0

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.PlayMode = "Snap To TimeSteps"
try:
    scene.Loop = 1
except Exception:
    pass
scene.AnimationTime = float((N - 1) // 2 if KIND == "flex" else (N - 1) // 4)   # open at peak bend
ResetCamera(view)
Render(view)
