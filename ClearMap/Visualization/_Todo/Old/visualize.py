# We will need some things from several places
from __future__ import division, absolute_import, print_function

from numpy.random import *  # for random sampling
import vispy
import vispy.scene
import vispy.visuals.tube as tb
from vesselVisual import VesselVisual
import numpy as np
from vispy import plot as vp
import Graph as gph

seed(42)

#######################################


def center(delta):
    delta = np.array(delta)

    def deltaDecal(x, **kwargs):
        return delta + np.mean(x, **kwargs)

    return deltaDecal


def plot3d(data):
    fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)
    vol_pw = fig[0, 0]
    vol_pw.volume(data)
    vol_pw.camera.elevation = 30
    vol_pw.camera.azimuth = 30
    vol_pw.camera.scale_factor /= 1.5

    fig.show(run=True)


def plotLines(graph, color=(0.8, 0.1, 0.1, 0.25), center=np.mean, view=None):
    # extract graph data
    pos = graph.vertexCoordinates()
    if center is not None:
        pos = pos - center(pos, axis=0)

    connect = np.asarray(graph.edges(), dtype=np.int32)

    # build visuals
    GraphPlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.LineVisual)
    if view is None:
        # build canvas
        canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 85
        view.camera.distance = 0
        view.camera.elevation = 0
        view.camera.azimuth = 0

    return GraphPlot3D(pos=pos, connect=connect,
                       width=2.0, color=color, parent=view.scene)


def plotVessels(graph, threshold=4, color=(0.8, 0.1, 0.1, 1), center=np.mean, view=None, reduced=False):
    """

    :param graph:
    :param threshold:
    :param center:
    :param view:
    :param reduced: If False, the tubes are made longer for better rendering
    :return:
    """
    # extract graph data
    pos = graph.vertexCoordinates()
    if center is not None:
        pos = pos - center(pos, axis=0)
    radii = graph.edgeRadius()
    radii /= 600  # 216  # Normalize radii

    connect = np.asarray(graph.edges(), dtype=np.int32)

    # build visuals
    CreateVessels = vispy.scene.visuals.create_visual_node(VesselVisual)

    if view is None:
        # build canvas
        canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 100
        view.camera.distance = 0
        view.camera.elevation = 0
        view.camera.azimuth = 0

    return CreateVessels(pos, connect, radii, color=color, threshold=threshold, parent=view.scene, reduced=reduced)


def get_view():
    canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 100
    view.camera.distance = 0
    view.camera.elevation = 0
    view.camera.azimuth = 0

    canvas.unfreeze()
    canvas.view = canvas.central_widget.add_view()

    return view

if __name__ == '__main__':

    ######################################
    # Plot data in 3D
    ######################################

    arr = np.load("Data/debug.npy")
    arr = arr[500:800, 500:800, 500:800]
    plot3d(arr)

    ######################################
    # Plot vessels
    ######################################

    graph = gph.load("Data/comparaison/podo-igg_graph_full.gt")
    plotVessels(graph, threshold=6)
    vispy.app.run()

    ######################################
    # Plot lines
    ######################################

    graph = gph.load("Data/comparaison/podo-igg_graph_full.gt")
    plotLines(graph)
    vispy.app.run()

    ######################################

    ######################################
    # Plot lines and vessels
    ######################################

    graph = gph.load("Data/comparaison/podo-igg_graph_full.gt")

    # Extract subgraph
    # condition = np.array([600, 500, 100])
    # prop = (graph.vertexCoordinates() > condition).all(axis=1)
    # graph = graph.subGraph(vertexFilter=prop)

    view = get_view()

    plotVessels(graph, view=view, color=(0.8, 0.1, 0.1, 1), center=center([0, 0, 0]))
    plotLines(graph, view=view, color=(0.8, 0.1, 0.1, 1), center=center([0, 0, 0]))
    vispy.app.run()

    ######################################
    # Compare two graphs
    ######################################

    graph1 = gph.load("Data/comparaison/crop-igg_graph_full.gt")
    graph2 = gph.load("Data/comparaison/crop-podo_graph_full.gt")

    view = get_view()

    plotVessels(graph1, view=view, color=(0.8, 0.1, 0.1, 0.25), center=center([0, 0, 0]))
    plotLines(graph1, view=view, color=(0.8, 0.1, 0.1, 0.25), center=center([0, 0, 0]))
    plotVessels(graph2, view=view, color=(0.1, 0.8, 0.1, 0.25), center=center([0, 0, 0]))
    plotLines(graph2, view=view, color=(0.1, 0.8, 0.1, 0.25), center=center([0, 0, 0]))
    vispy.app.run()
