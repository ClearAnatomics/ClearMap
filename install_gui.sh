#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

config_folder="$HOME/.clearmap"

eval "$(conda shell.bash hook)"  # Required to activate conda envs


# Verify CUDA is functional
python -c 'import os, sys; sys.path.append(os.getcwd()); \
from ClearMap.Utils.install_utils import assert_cuda; assert_cuda()' || exit 1

# Amend environment file for compatibility with installed CUDA version
python -c 'import os, sys; sys.path.append(os.getcwd()); \
from ClearMap.Utils.install_utils import patch_cuda_toolkit_version;\
 patch_cuda_toolkit_version(os.path.join(os.getcwd(), "ClearMapUi.yml"))'

# Create ClearMapUi env if not present and activate
conda env list | grep ClearMapUi
if [ $? -eq 1 ]; then
  conda env create -f "$BASEDIR/ClearMapUi.yml"
fi
conda activate ClearMapUi

# Install ClearMap
python "setup.py" install

# Create config folder if missing
if [ ! -d "$config_folder" ]; then
  mkdir "$config_folder"
fi

# Install or update ClearMap config
cd "$HOME" || exit  # Should not be in same folder to import from installed version
python -m ClearMap.config.update_config
#python ClearMap/Environment.py  # Makes sure everything is compiled


# TODO: Prompt for env variables: (tmp, elastix ...)

# CONFIG
clearmap_install_path=$(python -c 'from ClearMap.config.update_config import CLEARMAP_DIR; print(CLEARMAP_DIR)')
if [ "$clearmap_install_path" == "" ];then
    echo "ERROR: could not get ClearMap install path"
    exit 1
fi
echo "ClearMap installed at \"$clearmap_install_path\""

# Create Linux desktop menus
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

conda shell.bash hook >> "$clearmap_install_path/conda_init.sh"
chmod u+x "$clearmap_install_path/conda_init.sh"
chmod u+x "$clearmap_install_path/start_gui.sh"

chmod u+x "$clearmap_install_path/ClearMap/External/elastix/build/bin/*"

# config env for shipped Elastix binary
conda env config vars set "LD_LIBRARY_PATH=$clearmap_install_path/ClearMap/External/elastix/build/bin/:$LD_LIBRARY_PATH"

echo "
ClearMapUi installed
To use it, open a terminal and run:
    conda activate ClearMapUi
    clear_map_ui
Alternatively, use the start menu menu_entry
"

conda deactivate
