import os.path
import shutil
import sys

from tqdm import tqdm

from ClearMap.Scripts.cell_map import CellDetector
from ClearMap.Scripts.sample_preparation import PreProcessor
from ClearMap.config.config_loader import get_configs, ConfigLoader


def backup_file(file_path):  # REFACTOR: put in workspace or IO
    base_path, ext = os.path.splitext(file_path)
    new_path = base_path + '.bcp' + ext
    shutil.copy(file_path, new_path)


def process_folders(folders):
    for folder in tqdm(folders, desc='Processing sample ', unit='brain'):
        cfg_loader = ConfigLoader(folder)
        configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))

        pre_proc = PreProcessor()
        pre_proc.setup(configs)
        # print('Setting up atlases')
        pre_proc.setup_atlases()

        cell_detector = CellDetector(pre_proc)

        backup_file(cell_detector.workspace.filename('cells'))
        cell_detector.atlas_align()

        backup_file(cell_detector.workspace.filename('cells', extension='csv'))
        cell_detector.export_as_csv()


def main():
    with open(sys.argv[1], 'r') as infile:
        folders = infile.readlines()
    folders = [f.strip() for f in folders if not f.startswith('#')]
    process_folders(folders)


if __name__ == '__main__':
    main()
