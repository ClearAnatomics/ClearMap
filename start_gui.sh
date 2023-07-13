#!/bin/bash

clearmap_egg_path=$1

#config_folder="$HOME/.clearmap"

log_path=/tmp/clearmap_startup.log

rm "$log_path"
echo 'Starting' >> "$log_path"

source "$clearmap_egg_path/conda_init.sh"  # Required to activate conda envs
echo 'Conda initialised' >> "$log_path"

conda env list | grep ClearMapUi
if [ $? -eq 1 ]; then
  echo 'Conda command failed' >> "$log_path"
  exit "1"  #  TODO: see
fi
conda activate ClearMapUi
echo 'Conda Activated' >> "$log_path"

clear_map_ui
echo 'UI started' >> "$log_path"

conda deactivate
echo "Process finished" >> "$log_path"
