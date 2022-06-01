eval "$(conda shell.bash hook)"

conda env list | grep ClearMapUi
if [ $? -eq 1 ]; then
  conda env create -f ClearMapUi.yml
fi
conda activate ClearMapUi

conda deactivate