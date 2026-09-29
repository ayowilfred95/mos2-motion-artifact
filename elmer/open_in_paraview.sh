#!/bin/bash
# Open an Elmer result in ParaView on the Screen Sharing display.
# usage (on the Mac): elmer/open_in_paraview.sh <model folder name> [WARP] [YSCALE] [LIM] ["title"] [script]
#   wrist animation:  elmer/open_in_paraview.sh wrist_motion 20 1 0.06 "" show_motion.py
M=${1:-baseline}; W=${2:-20}; Y=${3:-1}; L=${4:-0.06}; T=${5:-$M}; S=${6:-show_result.py}
orb -m elmer bash -c "
export DISPLAY=:1 WARP=$W YSCALE=$Y LIM=$L TITLE='$T'
pkill -x paraview; sleep 2
cd /Users/ayomide/Downloads/mos2_motion_artifact/elmer/models/$M || exit 1
nohup paraview --script=/Users/ayomide/Downloads/mos2_motion_artifact/elmer/$S > /tmp/pv_run.log 2>&1 &
for i in \$(seq 1 60); do WIN=\$(xdotool search --name 'Welcome to ParaView' 2>/dev/null | head -1); [ -n \"\$WIN\" ] && break; sleep 1; done
sleep 3
[ -n \"\$WIN\" ] && xdotool windowactivate --sync \$WIN 2>/dev/null && xdotool key Escape 2>/dev/null
sleep 2
wmctrl -r 'ParaView 5.11.2' -b add,maximized_vert,maximized_horz; wmctrl -a 'ParaView 5.11.2'
"
