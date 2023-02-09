#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
if [ "$1" == "" ]; then
    ENV_FILE_PATH="ClearMapUi.yml"
else
    ENV_FILE_PATH=$1
fi

function red(){  #  From https://stackoverflow.com/a/57096493
    echo -e "\x1B[31m $1 \x1B[0m"
    if [ -n "${2}" ]; then
        echo -e "\x1B[31m $($2) \x1B[0m"
    fi
}
function green(){
    echo -e "\x1B[32m $1 \x1B[0m"
    if [ -n "${2}" ]; then
        echo -e "\x1B[32m $($2) \x1B[0m"
    fi
}


config_folder="$HOME/.clearmap"
prep_python="import os, sys; sys.path.append(os.getcwd());"

eval "$(conda shell.bash hook)"  # Required to activate conda envs

echo "To speed up solving the environment,
we recommend using the experimental libmamba solver for conda"
read -r -p "Do you wish to install this program ([y]/n)?" answer
case "$answer" in
    [nN][oO]|[nN])
        green "Using default solver";
        mamba_installed=false;
        ;;
    *)
        green "Using libmamba";
        conda install -n base conda-libmamba-solver; mamba_installed=true;
        ;;
esac


# Verify CUDA is functional
echo "Checking CUDA installation"
python -c "$prep_python \
from ClearMap.Utils.install_utils import assert_cuda; assert_cuda()" || exit 1
green "OK"

# Amend environment file for compatibility with installed CUDA version
echo "Updating CUDA dependencies for ClearMap"
echo "Creating temporary environment"
if [ "$mamba_installed" = true ] ; then
    conda create -n clearmap_tmp_env python pyyaml --experimental-solver=libmamba || exit 1
else
    conda create -n clearmap_tmp_env python pyyaml || exit 1
fi
conda activate clearmap_tmp_env || exit 1
green "Done"

echo "Getting env name"
ENV_NAME=$(python -c "from ClearMap.Utils.install_utils import get_env_name; \
env_name=get_env_name('$BASEDIR/$ENV_FILE_PATH'); print(env_name)")
green "Env name: $ENV_NAME"

echo "Patching environment file"
python -c "$prep_python \
from ClearMap.Utils.install_utils import patch_cuda_toolkit_version, patch_pytorch_cuda_version; \
patch_cuda_toolkit_version(os.path.join(os.getcwd(), '$ENV_FILE_PATH'), 'tmp_env_file.yml'); \
patch_pytorch_cuda_version(os.path.join(os.getcwd(), '$ENV_FILE_PATH'), 'tmp_env_file.yml')" || exit 1
green "Done"
conda deactivate
conda env remove -n clearmap_tmp_env

# Create environment if not present, otherwise update the packages and activate
echo "Checking ClearMap env"
conda env list | grep "$ENV_NAME"
if [ $? -eq 1 ]; then
    green "$ENV_NAME not found, creating env"
    if [ "$mamba_installed" = true ] ; then
        conda env create -f "$BASEDIR/tmp_env_file.yml" --experimental-solver=libmamba || exit 1
    else
        conda env create -f "$BASEDIR/tmp_env_file.yml" || exit 1
    fi
else
    green "Found $ENV_NAME, updating env"
    if [ "$mamba_installed" = true ] ; then
        conda env update --name "$ENV_NAME" --file "$BASEDIR/tmp_env_file.yml" --experimental-solver=libmamba || exit 1 # --prune
    else
        conda env update --name "$ENV_NAME" --file "$BASEDIR/tmp_env_file.yml" || exit 1 # --prune
    fi
fi
conda activate "$ENV_NAME" || exit 1

# Install ClearMap
echo "Installing"
python "setup.py" install || exit 1
echo "Done"
echo "Checking pytorch installation"
python -c "$prep_python \
from ClearMap.Utils.install_utils import check_pytorch; check_pytorch()" || exit 1
green "Pytorch installed and functional with CUDA support"

# Create config folder if missing
if [ ! -d "$config_folder" ]; then
   mkdir "$config_folder" || exit 1
fi

# Install or update ClearMap config
cd "$HOME" || exit 1 # Should not be in same folder to import from installed version
python -m ClearMap.config.update_config  || exit 1
#python ClearMap/Environment.py  # Makes sure everything is compiled


# TODO: Prompt for env variables: (tmp, elastix ...)

# CONFIG
clearmap_install_path=$(python -c "from ClearMap.config.update_config import CLEARMAP_DIR; print(CLEARMAP_DIR)")
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

desktop_file="$HOME/.local/share/applications/menulibre-clearmap2.desktop"  # TODO: check if folder exists
echo "$menu_entry" > "$desktop_file" && echo "wrote $desktop_file"

desktop_file="$HOME/.gnome/apps/menulibre-clearmap2.desktop"  # TODO: check if folder exists
echo "$menu_entry" > "$desktop_file" && echo "wrote $desktop_file"

conda shell.bash hook >> "$clearmap_install_path/conda_init.sh" || exit 1
chmod u+x "$clearmap_install_path/conda_init.sh" || exit 1
chmod u+x "$clearmap_install_path/start_gui.sh" || exit 1

chmod u+x "$clearmap_install_path/ClearMap/External/elastix/build/bin/"* || exit 1

# Configure environment to amend LD_LIBRARY_PATH to point to custom Elastix binary shipped with ClearMap
conda env config vars set "LD_LIBRARY_PATH=$clearmap_install_path/ClearMap/External/elastix/build/bin/:$LD_LIBRARY_PATH" || exit 1

green "
$ENV_NAME installed
To use it, open a terminal and run:
    conda activate $ENV_NAME
    clear_map_ui
Alternatively, use the ClearMap entry in the start menu
"

conda deactivate
