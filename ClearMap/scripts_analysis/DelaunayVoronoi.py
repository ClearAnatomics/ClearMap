#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def voronoi(P):
    delauny = Delaunay(P)
    triangles = delauny.points[delauny.vertices]

    lines = []

    # Triangle vertices
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    lines.extend(zip(A, B))
    lines.extend(zip(B, C))
    lines.extend(zip(C, A))
    lines = matplotlib.collections.LineCollection(lines, color='r')
    plt.gca().add_collection(lines)

    circum_centers = np.array([triangle_csc(tri) for tri in triangles])

    segments = []
    for i, triangle in enumerate(triangles):
        circum_center = circum_centers[i]
        for j, neighbor in enumerate(delauny.neighbors[i]):
            if neighbor != -1:
                segments.append((circum_center, circum_centers[neighbor]))
            else:
                ps = triangle[(j+1)%3] - triangle[(j-1)%3]
                ps = np.array((ps[1], -ps[0]))

                middle = (triangle[(j+1)%3] + triangle[(j-1)%3]) * 0.5
                di = middle - triangle[j]

                ps /= np.linalg.norm(ps)
                di /= np.linalg.norm(di)

                if np.dot(di, ps) < 0.0:
                    ps *= -1000.0
                else:
                    ps *= 1000.0
                segments.append((circum_center, circum_center + ps))
    return segments

def triangle_csc(pts):
    rows, cols = pts.shape

    A = np.bmat([[2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                 [np.ones((1, rows)), np.zeros((1, 1))]])

    b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
    x = np.linalg.solve(A,b)
    bary_coords = x[:-1]
    return np.sum(pts * np.tile(bary_coords.reshape((pts.shape[0], 1)), (1, pts.shape[1])), axis=0)

if __name__ == '__main__':
    P = np.random.random((300,2))

    X,Y = P[:,0],P[:,1]

    fig = plt.figure(figsize=(4.5,4.5))
    axes = plt.subplot(1,1,1)


    plt.axis([-0.05,1.05,-0.05,1.05])

    segments = voronoi(P)
    lines = matplotlib.collections.LineCollection(segments, color='k')
    axes.add_collection(lines)
    plt.axis([-0.05,1.05,-0.05,1.05])
    plt.scatter(X, Y, marker='.', color='g')
    segments = voronoi(P)
    plt.show()

    P2 = np.array(segments).reshape(1743 * 2, 2)
    fig = plt.figure(figsize=(4.5, 4.5))
    axes = plt.subplot(1, 1, 1)

    plt.axis([-0.05, 1.05, -0.05, 1.05])

    segments2 = voronoi(P2)
    lines2 = matplotlib.collections.LineCollection(segments2, color='g')
    axes.add_collection(lines2)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    # plt.scatter(X, Y, marker='.', color='g')
    segments = voronoi(P)
    plt.show()