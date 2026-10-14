#!/bin/bash
# usage: qa.sh NAME...  -> assets/sam3d/NAME/contact.png
D=~/projects/yam-arcade/assets/sam3d/_work
~/projects/yam-arcade/.venvs/assets/bin/python -I $D/scripts/contact.py section "$@" 2>&1 | grep -v Deprec
cd $D && blender -b -P $D/scripts/blender_views.py -- $D/qa "$@" > $D/qa/blender.log 2>&1 || tail -20 $D/qa/blender.log
~/projects/yam-arcade/.venvs/assets/bin/python -I $D/scripts/contact.py tile "$@"
