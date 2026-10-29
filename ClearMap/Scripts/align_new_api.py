from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


def plot_registration_results(aligner: RegistrationProcessor, channel: str, composite: bool = False):
    dvs, titles = aligner.plot_registration_results(channel=channel, composite=composite)
    if not composite:
        link_dataviewers_cursors(dvs)
