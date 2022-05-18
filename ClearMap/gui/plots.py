from ClearMap.gui.widgets import RedCross


def link_dataviewers_cursors(dvs):  # TODO: move to DataViewer module
    for i, dv in enumerate(dvs):
        cross = RedCross()
        dv.view.addItem(cross)
        dv.cross = cross
        pals = dvs.copy()
        pals.pop(i)
        dv.pals = pals