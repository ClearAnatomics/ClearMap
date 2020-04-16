from __future__ import division

from vispy.visuals.mesh import MeshVisual
import numpy as np
from numpy.linalg import norm
from vispy.util.transforms import rotate
from vispy.color import ColorArray
from tqdm import tqdm


class VesselVisual(MeshVisual):
    """Displays vessels with tubes in order to render the radius.

    Each

    The tube mesh is corrected following its Frenet curvature and
    torsion such that it varies smoothly along the curve, including if
    the tube is closed.

    Parameters
    ----------
    points : ndarray
        An array of (x, y, z) points describing the path along which the
        tube will be extruded.
    radius : float
        The radius of the tube. Defaults to 1.0.
    closed : bool
        Whether the tubes should be closed, joining the last point to the
        first. Defaults to False.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section. Defaults to 8.
    shading : str | None
        Same as for the `MeshVisual` class. Defaults to 'smooth'.
    vertex_colors: ndarray | None
        Same as for the `MeshVisual` class.
    face_colors: ndarray | None
        Same as for the `MeshVisual` class.
    mode : str
        Same as for the `MeshVisual` class. Defaults to 'triangles'.

    """

    def __init__(self, positions, connect, radii,
                 closed=False,
                 tube_points=8,
                 shading='smooth',
                 color=(0.8, 0.1, 0.1, 1),
                 threshold=4,
                 reduced=False,
                 mode='triangles'):

        all_indices = []
        all_vertices = []
        cpt = 0

        color1 = np.array(color)
        color2 = np.array((0.1, 0.1, 0.8, 1))
        color_array = []

        Lambda = 2 # How much the tube is longer than the edge

        for k in tqdm(range(len(connect))):

            p1 = connect[k, 0]  # first point
            p2 = connect[k, 1]  # second point
            points = np.concatenate((positions[p1][None, :], positions[p2][None, :]), axis=0)

            radius = radii[k]

            if radius > 0 and not np.array_equal(points[0], points[1]):
                # Attention des erreurs, certains point sont en double

                tangents, normals, binormals = _frenet_frames(points, closed)
                segments = len(points) - 1

                # make the tube longer than the edge
                if not reduced:
                    mid_point = np.mean(points, axis=0)
                    points[0] += (Lambda-1) * (points[0] - mid_point)
                    points[1] += (Lambda-1) * (points[1] - mid_point)

                # get the positions of each vertex
                grid = np.zeros((len(points), tube_points, 3))

                for i in range(len(points)):
                    pos = points[i]
                    normal = normals[i]
                    binormal = binormals[i]

                    # Add a vertex for each point on the circle
                    v = np.arange(tube_points,
                                  dtype=np.float) / tube_points * 2 * np.pi
                    cx = -1. * radius * np.cos(v)
                    cy = radius * np.sin(v)
                    grid[i] = (pos + cx[:, np.newaxis] * normal +
                               cy[:, np.newaxis] * binormal)

                # construct the mesh
                for i in range(segments):
                    for j in range(tube_points):

                        ip = (i + 1) % segments if closed else i + 1
                        jp = (j + 1) % tube_points

                        index_a = i * tube_points + j + cpt
                        index_b = ip * tube_points + j + cpt
                        index_c = ip * tube_points + jp + cpt
                        index_d = i * tube_points + jp + cpt

                        all_indices.append([index_a, index_b, index_d])
                        all_indices.append([index_b, index_c, index_d])

                        if radius < threshold:
                            color_array.append(color1)
                            color_array.append(color1)
                        else:
                            color_array.append(color2)
                            color_array.append(color2)

                vertices = grid.reshape(grid.shape[0] * grid.shape[1], 3)
                cpt += 2 * segments * tube_points
                all_vertices.append(vertices)

        # Format arrays to be fed to MeshVisual
        all_vertices = np.concatenate(all_vertices)
        all_indices = np.array(all_indices, dtype=np.uint32)
        color_array = np.array(color_array)

        MeshVisual.__init__(self, all_vertices, all_indices,
                            # vertex_colors=color,
                            face_colors=color_array,
                            shading=shading,
                            mode=mode)


def _frenet_frames(points, closed):
    '''Calculates and returns the tangents, normals and binormals for
    the tube.'''
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    if not closed:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    mags = np.sqrt(np.sum(tangents * tangents, axis=1))

    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)

    normals[0] = np.cross(tangents[0], vec)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i - 1]

        vec = np.cross(tangents[i - 1], tangents[i])
        if norm(vec) > epsilon:
            vec /= norm(vec)
            theta = np.arccos(np.clip(tangents[i - 1].dot(tangents[i]), -1, 1))
            normals[i] = rotate(-np.degrees(theta),
                                vec)[:3, :3].dot(normals[i])

    if closed:
        theta = np.arccos(np.clip(normals[0].dot(normals[-1]), -1, 1))
        theta /= len(points) - 1

        if tangents[0].dot(np.cross(normals[0], normals[-1])) > 0:
            theta *= -1.

        for i in range(1, len(points)):
            normals[i] = rotate(-np.degrees(theta * i),
                                tangents[i])[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals
