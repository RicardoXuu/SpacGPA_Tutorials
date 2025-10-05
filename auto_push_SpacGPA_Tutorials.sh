#!/bin/bash

#
current_time=$(date "+%Y-%m-%d %H:%M:%S")

# 
cd /dta/ypxu/SpacGPA/Dev_Version/SpacGPA_Tutorials

# 
if git diff-index --quiet HEAD --; then
    echo "[$current_time] No changes to commit." >> auto_push_SpacGPA_Tutorials.log
    exit 0
fi

#
git add .

# 
git commit -m "code update at $current_time"

