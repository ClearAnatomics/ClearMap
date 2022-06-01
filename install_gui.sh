#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

config_folder="$HOME/.clearmap"

eval "$(conda shell.bash hook)"  # Required to activate conda envs

conda env list | grep ClearMapUi
if [ $? -eq 1 ]; then
  conda env create -f "$BASEDIR/ClearMapUi.yml"
fi
conda activate ClearMapUi

python "setup.py" install

if [ ! -d "$config_folder" ]; then
  mkdir "$config_folder"
fi

cd "$HOME" || exit  # Should not be in same folder to import from installed version
python -m ClearMap.config.update_config
#python ClearMap/Environment.py  # Makes sure everything is compiled


# TODO: Prompt for env variables: (tmp, elastix ...)

# CONFIG
clearmap_install_path=$(python -c 'from ClearMap.config.update_config import CLEARMAP_DIR; print(CLEARMAP_DIR)')
echo "ClearMap installed at \"$clearmap_install_path\""

menu_entry="[Desktop Entry]
Version=1.1
Type=Application
Name=ClearMap2
Comment=The ClearMap2 Pipeline GUI
Icon=$clearmap_install_path/ClearMap/gui/icons/logo_cyber.png
Exec=$clearmap_install_path/start_gui.sh $clearmap_install_path
Actions=
Categories=Biology;Education;X-XFCE;X-Xfce-Toplevel;
StartupNotify=true"

desktop_file="$HOME/.local/share/applications/menulibre-clearmap2.desktop"
echo "$menu_entry" > "$desktop_file"
echo "wrote $desktop_file"

desktop_file="$HOME/.gnome/apps/menulibre-clearmap2.desktop"
echo "$menu_entry" > "$desktop_file"
echo "wrote $desktop_file"

#echo "Entry:
#$menu_entry"

conda shell.bash hook >> "$clearmap_install_path/conda_init.sh"


echo "
ClearMapUi installed
To use it, open a terminal and run:
    clear_map_ui
Alternatively, add a menu entry for the file 'start_gui.sh'"

conda deactivate