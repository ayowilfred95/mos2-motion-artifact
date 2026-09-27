"""
ParaView startup script: show an Elmer result as a coloured axial-strain map.

  paraview --script=show_result.py        (run inside the model folder, e.g. elmer/models/baseline)
Opens results/*_t0001.vtu, colours it by axial strain (%), and warps it by the displacement
(exaggerated so the bending is visible on a 0.2 mm slice).
"""
import glob
import os
from paraview.simple import *  # noqa: F401,F403

WARP = float(os.environ.get("WARP", "20"))        # visual exaggeration of the displacement
YSCALE = float(os.environ.get("YSCALE", "1"))     # stretch of the thickness direction (display only)
LIM = float(os.environ.get("LIM", "0.06"))        # colour range +-LIM (%)
TITLE = os.environ.get("TITLE", "Baseline sensor stack bent to wrist curvature (R = 25 mm)")

f = sorted(glob.glob(os.path.join("results", "*_t0001.vtu")))[0]
src = XMLUnstructuredGridReader(registrationName=os.path.basename(f), FileName=[f])
calc = Calculator(registrationName="axial strain (%)", Input=src)
calc.ResultArrayName = "axial strain (%)"
calc.Function = "strain_xx*100"
warp = WarpByVector(registrationName=f"bent shape (x{WARP:g})", Input=calc)
warp.Vectors = ["POINTS", "displacement"]
warp.ScaleFactor = WARP

view = GetActiveViewOrCreate("RenderView")
disp = Show(warp, view)
ColorBy(disp, ("POINTS", "axial strain (%)"))
lut = GetColorTransferFunction("axialstrain")
lut.ApplyPreset("Cool to Warm", True)
lim = LIM                                    # symmetric colour range (%), white = zero strain
lut.RescaleTransferFunction(-lim, lim)
GetOpacityTransferFunction("axialstrain").RescaleTransferFunction(-lim, lim)
line = Contour(registrationName="neutral plane (strain = 0)", Input=warp)
line.ContourBy = ["POINTS", "axial strain (%)"]
line.Isosurfaces = [0.0]
ldisp = Show(line, view)
ColorBy(ldisp, None)
ldisp.AmbientColor = ldisp.DiffuseColor = [0.0, 0.0, 0.0]
ldisp.LineWidth = 3.0
bar = GetScalarBar(lut, view)
bar.Title = "axial strain (%)"
bar.ComponentTitle = ""
bar.Visibility = 1
view.InteractionMode = "2D"
view.OrientationAxesVisibility = 0
text = Text(registrationName="title")
extra = f", thickness stretched x{YSCALE:g}" if YSCALE != 1 else ""
text.Text = (TITLE + "\nred = stretched, blue = compressed, black line = neutral plane (zero strain); "
             f"bend exaggerated x{WARP:g}{extra}")
tdisp = Show(text, view)
tdisp.WindowLocation = "Upper Center"
tdisp.FontSize = 14
for dsp in (disp, ldisp):
    dsp.Scale = [1.0, YSCALE, 1.0]
Hide(src, view)
Hide(calc, view)
ResetCamera(view)
Render(view)
