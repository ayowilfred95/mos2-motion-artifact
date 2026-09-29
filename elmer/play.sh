#!/bin/bash
# Play an Elmer bending motion in ParaView (Screen Sharing window).
# usage:  elmer/play.sh wrist | elbow | walking | pet | island
case "$1" in
  wrist)   M=wrist_motion;        V="KAPPA_PEAK=40 KIND=flex PERIOD=2.0 T_SUB=25e-6 WARP=20 YSCALE=1" ;;
  elbow)   M=elbow_motion;        V="KAPPA_PEAK=25 KIND=flex PERIOD=2.0 T_SUB=25e-6 WARP=20 YSCALE=1" ;;
  walking) M=walking_motion;      V="KAPPA_PEAK=3.3333333 KIND=osc PERIOD=0.5555556 T_SUB=25e-6 WARP=150 YSCALE=1" ;;
  pet)     M=pet_wrist_motion;    V="KAPPA_PEAK=40 KIND=flex PERIOD=2.0 T_SUB=2.5e-6 WARP=20 YSCALE=8" ;;
  island)  M=island_wrist_motion; V="KAPPA_PEAK=40 KIND=flex PERIOD=2.0 T_SUB=25e-6 WARP=1 YSCALE=8" ;;
  *) echo "usage: elmer/play.sh wrist | elbow | walking | pet | island"; exit 1 ;;
esac
TITLE=$(grep -A4 "\"$M\"" "$(dirname "$0")/data/motions.json" | grep '"title"' | sed 's/.*"title": "\(.*\)",*/\1/')
orb -m elmer bash -c "
export DISPLAY=:1 $V TITLE='$TITLE (21 solved steps)'
pkill -x paraview; sleep 2
cd /Users/ayomide/Downloads/mos2_motion_artifact/elmer/models/$M || exit 1
nohup paraview --script=/Users/ayomide/Downloads/mos2_motion_artifact/elmer/show_motion.py > /tmp/pv_run.log 2>&1 &
for i in \$(seq 1 60); do WIN=\$(xdotool search --name 'Welcome to ParaView' 2>/dev/null | head -1); [ -n \"\$WIN\" ] && break; sleep 1; done
sleep 3
[ -n \"\$WIN\" ] && xdotool windowactivate --sync \$WIN 2>/dev/null && xdotool key Escape 2>/dev/null
sleep 2
wmctrl -r 'ParaView 5.11.2' -b add,maximized_vert,maximized_horz; wmctrl -a 'ParaView 5.11.2'
" 2>&1 | grep -v "BadWindow\|X Error\|Major opcode\|Resource id\|Serial number"
echo "Opened $1 in ParaView. Press the green Play button."
