"""
ParaView startup script: play an Elmer bending motion (one result file per step) as an animation, with a
live readout of the MoS2 strain computed by Elmer and the cortisol the sensor would report.

  paraview --script=show_motion.py        (run inside elmer/models/wrist_motion)

The readout: channel strain = eta x (Elmer strain at the MoS2 mid-plane, x = 0), then the sensor equations of
motion_artifact_sim.py (gate-dependent gauge factor, Schottky contacts, cross-talk, Langmuir binding at
equilibrium, Debye screening) at the true cortisol level TRUE_NGML.
"""
import glob
import math
import os
from paraview.simple import *  # noqa: F401,F403

WARP = 20.0            # visual exaggeration of the bend
TRUE_NGML = 30.0       # true sweat cortisol (ng/mL)
ETA = 0.8              # vdW strain transfer, Al2O3-capped MoS2
Y_MOS2 = 25.0e-6 + 50e-9 + 0.65e-9 / 2
DT, KAPPA_PEAK, N = 0.1, 40.0, 21

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
# two points inside the PI (y = 20 and 25 um, x = 0) and extended linearly to the MoS2 mid-plane: in bending
# the axial strain is linear through the thickness, so this is exact (difference to the direct Elmer
# SaveLine value < 0.1 %).
probe = PlotOverLine(registrationName="MoS2 channel probe", Input=src)
probe.Point1 = [1e-9, 20e-6, 0.0]
probe.Point2 = [1e-9, 25e-6, 0.0]
probe.Resolution = 1

# readout text, re-evaluated at every animation step (numbers inserted directly into the expression)
K = f"({KAPPA_PEAK!r}*0.5*(1-__import__('math').cos(2*__import__('math').pi*t/{DT * (N - 1)!r})))"          # applied bend curvature at time t
CAPP = f"(50.0*th/(1-th)/{NG!r})"                                      # apparent cortisol, ng/mL
expr = (
    f"(lambda t, eb: (lambda e: (lambda dI: (lambda th: "
    f"'time  %.1f s    bend curvature  %.1f 1/m  (radius %s)\\n"
    f"MoS2 strain (Elmer)  %.4f %%    current change from strain  %+.2f %%\\n"
    f"true cortisol  %.0f ng/mL    sensor reads  %s' % ("
    f"t, {K}, ('%.0f mm' % (1000/{K}) if {K} > 0.05 else 'flat'), e*100, dI*100, {TRUE_NGML!r}, "
    f"(('%.1f ng/mL  (error %+.1f)' % ({CAPP}, {CAPP} - {TRUE_NGML!r})) if th < 1 else 'OVER RANGE (saturated)')))"
    f"(({RTRUE!r}*(1+85*e) + dI)*{V!r}/{DVT!r}))"
    f"(1/(0.7*__import__('math').exp(-{GF!r}*e) + 0.3*__import__('math').exp(-5.5*e/0.025852)) - 1))"
    f"({ETA!r}*eb))({DT!r}*t_value, float(strain_xx[1]) + (float(strain_xx[1]) - float(strain_xx[0]))*{(Y_MOS2 - 25e-6) / 5e-6!r})"
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
title.Text = ("Elmer: baseline sensor during one wrist flexion (flat -> R = 25 mm -> flat, 21 solved steps)\n"
              f"red = stretched, blue = compressed, black line = neutral plane; bend exaggerated x{WARP:g}. "
              "Press the green Play button.")
tdisp = Show(title, view)
tdisp.WindowLocation = "Lower Center"
tdisp.FontSize = 13
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
scene.AnimationTime = 10.0            # open at full wrist bend
ResetCamera(view)
Render(view)
