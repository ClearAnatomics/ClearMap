from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin, QDesignerFormEditorInterface

from ClearMap.gui.spikes.triplet_control_widget import TripletControlWidget


class TripletControlPlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialized = False
        print(QLibraryInfo.pluginPath)

    def initialize(self, core: QDesignerFormEditorInterface) -> None:
        if self.initialized:
            return
        print('Initialising')
        self.initialized = True

    def createWidget(self, parent):
        return TripletControlWidget(parent=parent)

    def name(self):
        return "TripletControl"

    def group(self) -> str:
        return 'Input Widgets'

    def toolTip(self) -> str:
        return 'Widget for editing a group of 3 values'

    def includeFile(self):
        return "triplet_control_widget"

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="TripletControlWidget" name="TripletControlWidget">\n</widget>'