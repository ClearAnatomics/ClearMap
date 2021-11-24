# -*- coding: utf-8 -*-

from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *


class TubeMapWizardPlugin(QPyDesignerCustomWidgetPlugin):
    # The __init__() method is only used to set up the plugin and define its
    # initialized variable.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wizard_page_1 = QWizardPage()
        self.wizard_page_1.setObjectName(u"wizard_page_1")

        self.wizard_page_2 = QWizardPage()
        self.wizard_page_2.setObjectName(u"wizard_page_2")

        self.wizard_page_3 = QWizardPage()
        self.wizard_page_3.setObjectName(u"wizard_page_3")

        self.initialized = False

    # The initialize() and isInitialized() methods allow the plugin to set up
    # any required resources, ensuring that this can only happen once for each
    # plugin.
    def initialise(self, tube_map_wizard):
        if not tube_map_wizard.objectName():
            tube_map_wizard.setObjectName(u"tube_map_wizard")
        tube_map_wizard.resize(640, 480)

        tube_map_wizard.addPage(self.wizard_page_1)
        tube_map_wizard.addPage(self.wizard_page_2)
        tube_map_wizard.addPage(self.wizard_page_3)

        self.retranslate_ui(tube_map_wizard)

        QMetaObject.connectSlotsByName(tube_map_wizard)

    def isInitialized(self):
        return self.initialized

    def retranslate_ui(self, tube_map_wizard):
        tube_map_wizard.setWindowTitle(QCoreApplication.translate(
            "tube_map_wizard", u"TubeMap configuration wizard", None))

    # This factory method creates new instances of our custom widget with the
    # appropriate parent.
    def createWidget(self, parent):
        return TubeMapWizard(parent)

    # This method returns the name of the custom widget class that is provided
    # by this plugin.
    def name(self):
        return "TubeMapWizard"

    # Returns the name of the group in Qt Designer's widget box that this
    # widget belongs to.
    def group(self):
        return "ClearMap widgets"

    # Returns the icon used to represent the custom widget in Qt Designer's
    # widget box.
    def icon(self):
        return QIcon()

    # Returns a short description of the custom widget for use in a tool tip.
    def toolTip(self):
        return ""

    # Returns a short description of the custom widget for use in a "What's
    # This?" help message for the widget.
    def whatsThis(self):
        return ""

    # Returns True if the custom widget acts as a container for other widgets;
    # otherwise returns False. Note that plugins for custom containers also
    # need to provide an implementation of the QDesignerContainerExtension
    # interface if they need to add custom editing support to Qt Designer.
    def isContainer(self):
        return False

    # Returns an XML description of a custom widget instance that describes
    # default values for its properties. Each custom widget created by this
    # plugin will be configured using this description.
    def domXml(self):
        return '<widget class="TubeMapWizard" name="TubeMapWizard" />\n'

    # Returns the module containing the custom widget class. It may include
    # a module path.
    def includeFile(self):
        return "tubemap_wizard"
