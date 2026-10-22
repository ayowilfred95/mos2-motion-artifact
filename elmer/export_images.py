"""
Export publication strain-field images from the Elmer results (ParaView, 3200 x 1400 px = 600 dpi at
~13.5 cm width). Uses the same pipeline as show_result.py (strain map, neutral-plane line, colour bar).

  pvpython elmer/export_images.py        (inside the Linux machine; writes elmer/figures/elmer_strain_*.png;
                                          each image is rendered in its own pvpython process)
"""
import os
from paraview.simple import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = os.path.join(HERE, "show_result.py")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

IMAGES = [  # model folder, output name, WARP, YSCALE, colour range (%), title
    ("baseline", "baseline_bent", 20, 1, 0.06,
     "Baseline stack (25 um PI / 50 nm Au / 0.65 nm MoS2 / 5 nm Al2O3) at wrist curvature, R = 25 mm"),
    ("buried", "buried_channel", 20, 1, 0.06,
     "Buried channel (25 um PI / films / 23 um PI) at R = 25 mm: channel 0.98 um from the neutral plane"),
    ("pet_2p5", "pet_2p5um", 20, 8, 0.006,
     "Ultrathin 2.5 um PET substrate at R = 25 mm"),
    ("rigid_island", "rigid_island", 1, 8, 0.06,
     "Rigid island (3 mm x 20 um, 70 GPa) in 100 um PDMS under 25 um PI, wrist bending"),
]

def render(model, out, warp, ysc, lim, title):
    """One image per fresh pvpython process (a reused session keeps old sources and colour bars)."""
    os.chdir(os.path.join(HERE, "models", model))
    os.environ.update(WARP=str(warp), YSCALE=str(ysc), LIM=str(lim), TITLE=title)
    exec(open(SHOW).read(), {"__name__": "__main__"})
    view = GetActiveView()
    keep = ("bent shape", "neutral plane")   # no title inside the figure (JCE artwork rules)
    for (name, _), src in GetSources().items():
        if not name.startswith(keep):
            Hide(src, view)                      # the undeformed mesh and intermediate filters
    view.ViewSize = [4200, 1838]   # 600 dpi at 174 mm (JCE double-column width)
    view.OrientationAxesVisibility = 0
    bar = GetScalarBar(GetColorTransferFunction("axialstrain"), view)
    bar.TitleFontSize = 34
    bar.LabelFontSize = 29
    bar.ScalarBarLength = 0.45
    ResetCamera(view)
    view.CameraParallelScale *= 0.72
    path = os.path.join(FIGS, f"elmer_strain_{out}.png")
    SaveScreenshot(path, view, ImageResolution=[4200, 1838], TransparentBackground=0)
    print("wrote", path)


if __name__ == "__main__":
    import subprocess
    import sys
    if len(sys.argv) > 1:                         # child: render one image
        render(*IMAGES[int(sys.argv[1])])
    else:                                         # parent: one fresh process per image
        for i in range(len(IMAGES)):
            subprocess.run(["/usr/bin/pvpython", os.path.abspath(__file__), str(i)], check=True)
