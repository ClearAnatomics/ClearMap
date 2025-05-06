#!/usr/bin/env bash

PROG_NAME=$0

BASEDIR=$(dirname "$0")
if [ "$1" == "" ]; then
    ENV_FILE_PATH="ClearMapUi39.yml"
else
    ENV_FILE_PATH=$1
fi

usage() {
  cat << EOF >&2
  Usage: $PROG_NAME [-h] [-f <env-file-path>] [-s]

  --file, -f <env-file-path>: The environment file to use. Defaults to ClearMapUi.yml in the current folder
  --spyder, -s : Whether to install the appropriate spyder kernel
EOF
  exit 1
}

USE_SPYDER="False"
ENV_FILE_PATH=''
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--file)
      ENV_FILE_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--spyder)
      USE_SPYDER="True"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


# Get the current commit number and save it to a file
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    COMMIT_NUMBER=$(git rev-parse HEAD)
    if [ -d "ClearMap/config/" ]; then
        echo "commit_hash = \"$COMMIT_NUMBER\"" > ClearMap/config/commit_info.py
        echo "commit_date = \"$(git log -1 --format=%cd)\"" >> ClearMap/config/commit_info.py
        echo "branch = \"$(git rev-parse --abbrev-ref HEAD)\"" >> ClearMap/config/commit_info.py
        green "Commit number saved to commit_info.py"
    else
        red "Directory ClearMap/config/ does not exist. Skipping commit number save.
             for reference, the commit number is $COMMIT_NUMBER"
    fi
else
    yellow "Not a git repository. Skipping commit number save."
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

function yellow(){
    echo -e "\x1B[33m $1 \x1B[0m"
    if [ -n "${2}" ]; then
      echo -e "\x1B[33m $($2) \x1B[0m"
    fi
}

function green_n(){  # FIXME: parametrise above instead
    echo -n -e "\x1B[32m $1 \x1B[0m"
    if [ -n "${2}" ]; then
        echo -n -e "\x1B[32m $($2) \x1B[0m"
    fi
}

########################################################################################################################

green "Using env file $ENV_FILE_PATH"

if  [[ "$OSTYPE" == "linux-gnu"* ]]; then
    green "Checking dependencies"
    if [[ $(dpkg-query --show --showformat='${db:Status-Status}\n' 'build-essential') == "installed" ]]; then
        green "Compilation tools available"
    else
        red "Package \"build-essential\" was not found. It is required for compilation.
             Please install it using
             sudo apt install build-essential
             and try the installation process again"
             exit 1
    fi
fi

conda -V || { echo "Conda missing exiting"; exit 1; }
green "Conda installed and functional"


if [[ "$OSTYPE" == "darwin"* ]]; then
  green "MacOS was detected as your operating system.
   If you want to make full use of the parallel code in this program,
   we suggest you install the GCC compiler using homebrew."
  read -r -p "Do you wish to continue the installation process ([y]/n)?" answer
  case "$answer" in
    [nN][oO]|[nN])
        yellow "Aborting install";
        exit 0;
        ;;
    *)
        green "Continue install";
        ;;
esac
fi


config_folder="$HOME/.clearmap"
prep_python="import os, sys; sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'ClearMap'));"

eval "$(conda shell.bash hook)"  # Required to activate conda envs

echo "To speed up solving the environment,
we recommend using the experimental libmamba solver for conda"
read -r -p "Do you wish to install this program ([y]/n)?" answer
case "$answer" in
    [nN][oO]|[nN])
        green "Using default solver";
        solver_string="";
        ;;
    *)
        green "Using libmamba";
        conda install -y -n base conda-libmamba-solver;
        if conda install -h | grep -q "experimental-solver"; then
            solver_string="--experimental-solver=libmamba";  # Old conda
        else
            solver_string="--solver=libmamba";
        fi
        ;;
esac

# Amend environment file (notably for compatibility with installed CUDA version)
echo "Updating CUDA dependencies for ClearMap"
echo "  Creating temporary environment"
if  [[ "$OSTYPE" == "msys"* ]]; then
    conda create -y -n clearmap_tmp_env -c conda-forge python packaging pyyaml configobj "$solver_string"
else
    conda create -y -n clearmap_tmp_env -c conda-forge python packaging pyyaml configobj "$solver_string" || exit 1
fi
conda activate clearmap_tmp_env || exit 1
green "Done"


if  [[ "$OSTYPE" == "linux-gnu"* ]]; then  # FIXME: or "$OSTYPE" == "msys"
    green "ClearMap uses neural networks to perform vasculature analysis.
      The implementation of these networks relies on proprietary technology
      from nVIDIA called CUDA. To perform vasculature analysis, you will
      need a compatible graphics card and drivers."
    read -r -p "Do you wish to use this feature ([y]/n)?" answer
    case "$answer" in
        [nN][oO]|[nN])
            yellow "Skipping";
            USE_TORCH="False";
            ;;
        *)
            # Verify CUDA is functional
            green "Checking nVIDIA drivers and system CUDA installation";
            python -c "$prep_python import ClearMap; \
                       from ClearMap.Utils.install_utils import PytorchVersionManager; \
                       PytorchVersionManager.assert_cuda()" || exit 1
            green "OK";
            USE_TORCH="True";
            ;;
    esac
else
    USE_TORCH="False";
fi

pip_mode="True"
if [[ $USE_TORCH == "True" ]]; then
    green "Installing pytorch through conda may be restricted due to the license of the nvidia channel.
      If you prefer installing pytorch through pip, please select 'pip' below."
    read -r -p "Do you wish to install pytorch through conda (y/[n])?" answer
    case "$answer" in
        [yY][eE][sS]|[yY])
            pip_mode="False";
            ;;
        *)
            pip_mode="True";
            ;;
    esac
fi

echo "  Getting env name"
ENV_NAME=$(python -c "import os; from ClearMap.Utils.install_utils import EnvFileManager; \
env_mgr = EnvFileManager(os.path.normpath(os.path.join(os.getcwd(), '$ENV_FILE_PATH')), None); \
env_name=env_mgr.get_env_name(); print(env_name)")
green "Env name: $ENV_NAME"

echo "  Patching environment file"
green_n "ClearMap writes large amounts of data to the temporary folder of the system (~200GB).
If your system tmp folder is not located on a large of fast partition,
you can define an other path here. Default: /tmp"
read tmp_dir
if [ -z "$tmp_dir" ]; then
    tmp_dir="/tmp/"
fi
if [ ! -d "$tmp_dir" ]; then
    yellow "Folder missing $tmp_dir, it will be created"
    mkdir -p $tmp_dir || exit 1
fi

if  [[ "$OSTYPE" == "msys"* ]]; then
    export tmp_dir="C:/Users/$USERNAME/AppData/Local/Temp"
fi
green "Using temp folder: $tmp_dir"

python -c "$prep_python \
from ClearMap.Utils.install_utils import patch_env; \
patch_env(os.path.join(os.getcwd(), '$ENV_FILE_PATH'), 'tmp_env_file.yml', use_cuda_torch=$USE_TORCH, pip_mode=$pip_mode, use_spyder=$USE_SPYDER, tmp_dir='$tmp_dir')" || exit 1
conda deactivate
conda env remove -y -n clearmap_tmp_env
green "Done"

# Create environment if not present, otherwise update the packages and activate
echo "Checking ClearMap env"
conda env list | grep "$ENV_NAME"
if [ $? -eq 1 ]; then
    green "$ENV_NAME not found, creating env"
    conda env create -f "$BASEDIR/tmp_env_file.yml" $solver_string || exit 1
else
    green "Found $ENV_NAME, updating env"
    # TODO: See if --prune
    conda env update --name "$ENV_NAME" --file "$BASEDIR/tmp_env_file.yml" $solver_string || exit 1
fi
conda activate "$ENV_NAME" || exit 1
if [[ "$USE_TORCH" == "True" ]]; then
    echo "Checking pytorch installation"
    python -c "$prep_python \
    from ClearMap.Utils.install_utils import PytorchVersionManager; \
    PytorchVersionManager.check_pytorch()" && green "Pytorch installed and functional with CUDA support" || { red "Pytorch installation failed"; exit 1; }
fi

# Setup GCC for MaxOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -r -p "If GCC is installed on your system,
     type here the main version number.
      Otherwise, leave empty" answer

     re='^[0-9]+$'
    if ! [[ $answer =~ $re ]] ; then  # not a number
        yellow "No version number given, skipping GCC"
    else
        conda env config vars set "CC=gcc-$answer"
        conda env config vars set "CXX=g++-$answer"
        green "Using gcc and g++ v-$answer"
    fi
fi
# Install ClearMap
echo "Installing"
python "setup.py" install || exit 1
echo "Done"

# Create config folder if missing
green "Checking if ClearMap configuration directory exists at \"$config_folder\""
if [ ! -d "$config_folder" ]; then
    yellow "Config folder missing, creating it"
    mkdir "$config_folder" || exit 1
fi

# Install or update ClearMap config
srcdir=$(pwd)
cd "$HOME" || exit 1 # Exit source folder to import from installed version
green "Installing or updating ClearMap config"
python -m ClearMap.config.update_config  || exit 1
green "Done"

# TODO: Prompt for environment variables (elastix ...) to be set in env activate

# CONFIG
clearmap_install_path=$(python -c "from ClearMap.config.update_config import CLEARMAP_DIR; print(CLEARMAP_DIR)")
if [ "$clearmap_install_path" == "" ];then
    echo "ERROR: could not get ClearMap install path"
    exit 1
fi
echo "ClearMap installed at \"$clearmap_install_path\""

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Binary file support on your platform is currently not available
    Please specify the path to the elastix binary"
    read -r -p "Path to elastix binary: " elastix_path
    if [ ! -d "$elastix_path" ]; then
        red "Path to elastix binary not found"
        exit 1
    else
        conda activate "$ENV_NAME" || exit 1
        python -c "$prep_python \
        import ClearMap; \
        from ClearMap.Utils.install_utils import set_elastix_path; \
        set_elastix_path('$elastix_path')" || exit 1
    fi
fi

# Create Linux desktop menus
echo "Do you want to create a desktop menu entry.
Skip this if you are not running on linux or
running headless"
read -r -p "Create menu entry ([y]/n)?" answer
case "$answer" in  # FIXME:
    [nN][oO]|[nN])
        yellow "Skipping";
        ;;
    *)
        green "Creating entry";
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
        ;;
    *)
        green "Skipping menu entry";
        ;;
esac

chmod u+x "$clearmap_install_path/ClearMap/External/elastix/build/bin/"* || exit 1

# Configure environment to amend LD_LIBRARY_PATH to point to custom Elastix binary shipped with ClearMap
if  [[ "$OSTYPE" == "linux-gnu"* ]]; then
    lib_path_name="LD_LIBRARY_PATH"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    lib_path_name="DYLD_LIBRARY_PATH"
fi
conda env config vars set "$lib_path_name=$clearmap_install_path/ClearMap/External/elastix/build/bin/:$LD_LIBRARY_PATH" || exit 1

# FIXME: not for MaxOS

green "
$ENV_NAME installed
To use it, open a terminal and run:
    conda activate $ENV_NAME
    clearmap-ui
Alternatively, use the ClearMap entry in the start menu
"

# Cleanup
conda deactivate
cd "$srcdir" || exit 1
mv tmp_env_file.yml "${ENV_NAME}Real".yml
