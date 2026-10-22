#!/usr/bin/env bash
# Kill Strive and start it again, waiting until UE4SS has loaded the logger.
W="/c/Program Files (x86)/Steam/steamapps/common/GUILTY GEAR STRIVE/RED/Binaries/Win64"
taskkill //F //IM GGST-Win64-Shipping.exe >/dev/null 2>&1; taskkill //F //IM GGST.exe >/dev/null 2>&1
until ! tasklist | grep -q GGST; do sleep 1; done
rm -f "$W/UE4SS.log"
cmd //c start "" "steam://rungameid/1384160"
until grep -q "InputLogger\] UpdateBattle" "$W/UE4SS.log" 2>/dev/null; do sleep 2; done
