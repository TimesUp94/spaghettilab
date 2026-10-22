#!/usr/bin/env bash
# step.sh "<keys>" <wait_s> <shot_name> : press keys, wait, screenshot to $GGST_SHOTS (default $TEMP)
cd "$(dirname "$0")"
P=../../../.venv/Scripts/python.exe
S="${GGST_SHOTS:-$TEMP}"
[ -n "$1" ] && $P ggst.py keys "$1" ${4:-150}
$P -c "import time;time.sleep(${2:-1.5})"
$P ggst.py shot "$S/$3.png" && $P -c "import cv2;im=cv2.imread(r'$S/$3.png');cv2.imwrite(r'$S/$3.jpg',cv2.resize(im,(960,540)))"
