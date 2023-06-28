import numpy as np
import torch
import numpy as np
import time
from torch.autograd import Variable




def edgeOrientations(graph, normalize = False):
    vxyz = graph.vertex_coordinates();
    connectivity = graph.edge_connectivity()
    o = vxyz[connectivity[:,0]] - vxyz[connectivity[:,1]];
    if normalize:
        o = (o.T / np.linalg.norm(o, axis = 1)).T;
    return o;

def getNormalCoordinates(graph):
    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_coordinates()  # *1.625/25
    edges_centers = np.array([(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
    return edges_centers



def VarifoldProduct(sigma1, sigma2):
    oos2_1 = 1 / sigma1 ** 2
    oos2_2 = 1 / sigma2 ** 2

    def VProduct(x1, x2, or1, or2, rad1, rad2):
        K_loc_xy = torch.exp(-oos2_1 * torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2, dim=2))
        normalize = torch.sqrt(torch.sum(or1 ** 2, dim=1))[:, None] * torch.sqrt(torch.sum(or2 ** 2, dim=1))[None, :]
        K_or_xy = torch.sum(or1[:, None, :] * or2[None, :, :], dim=2) ** 2 / normalize
        # K_or_xy = 1

        K_rad_xy = torch.exp(-oos2_2 * torch.sum((rad1[:, None, :] - rad2[None, :, :]) ** 2, dim=2))
        tot=K_loc_xy * K_or_xy * K_rad_xy
        tot[torch.where(torch.isnan(tot))]=0
        return torch.sum(tot)

    return VProduct


def VarifoldDistance(sigma1, sigma2):
    VProd = VarifoldProduct(sigma1, sigma2)

    def VDist(x1, x2, or1, or2, rad1, rad2):
        return torch.sqrt(VProd(x1, x1, or1, or1, rad1, rad1)
                          - 2 * VProd(x1, x2, or1, or2, rad1, rad2)
                          + VProd(x2, x2, or2, or2, rad2, rad2))

    return VDist


def computeDistance(g1, g2, sigmaPos=1, sigmaRad=1):
    # get positions
    x1 = getNormalCoordinates(g1)
    x2 = getNormalCoordinates(g2)

    # get orientations
    or1 = edgeOrientations(g1,normalize=False)
    or2 = edgeOrientations(g2,normalize=False)

    # get radii
    rad1 = np.array(g1.edge_radii())[:, None]
    rad2 = np.array(g2.edge_radii())[:, None]

    # Convert everything to torch Variables
    x1 = Variable(torch.from_numpy(x1).float(), requires_grad=False)
    x2 = Variable(torch.from_numpy(x2).float(), requires_grad=False)
    or1 = Variable(torch.from_numpy(or1).float(), requires_grad=False)
    or2 = Variable(torch.from_numpy(or2).float(), requires_grad=False)
    rad1 = Variable(torch.from_numpy(rad1).float(), requires_grad=False)
    rad2 = Variable(torch.from_numpy(rad2).float(), requires_grad=False)
    # sigma1 = Variable(torch.FloatTensor(1, 1).fill_(sigmaPos), requires_grad=False)
    # sigma2 = Variable(torch.FloatTensor(1, 1).fill_(sigmaRad), requires_grad=False)

    Dist = VarifoldDistance(sigmaPos, sigmaRad)

    return float(Dist(x1, x2, or1, or2, rad1, rad2))




########################################################################################################################
# Convolutional Varifold Distance
########################################################################################################################


def VarifoldMatrices(sigma1, sigma2):
    oos2_1 = 1 / sigma1 ** 2
    oos2_2 = 1 / sigma2 ** 2

    def VMat(x1, x2, or1, or2, rad1, rad2):
        K_loc_xy = torch.exp(-oos2_1 * torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2, dim=2))
        normalize = torch.sqrt(torch.sum(or1 ** 2, dim=1))[:, None] * torch.sqrt(torch.sum(or2 ** 2, dim=1))[None, :]
        K_or_xy = torch.sum(or1[:, None, :] * or2[None, :, :], dim=2) ** 2 / normalize
        # K_or_xy = 1

        K_rad_xy = torch.exp(-oos2_2 * torch.sum((rad1[:, None, :] - rad2[None, :, :]) ** 2, dim=2))

        return K_loc_xy * K_or_xy * K_rad_xy

    return VMat


def ConvVarifoldDistance(sigmaPos, sigmaRad, sigmaVis, patchSize):
    VMat = VarifoldMatrices(sigmaPos, sigmaRad)

    def ConvVDist(x1, x2, or1, or2, rad1, rad2):
        mat11 = VMat(x1, x1, or1, or1, rad1, rad1)
        mat12 = VMat(x1, x2, or1, or2, rad1, rad2)
        mat22 = VMat(x2, x2, or2, or2, rad2, rad2)

        maxs1 = np.max(x1.data.numpy(), axis=0).astype(int)
        mins1 = np.min(x1.data.numpy(), axis=0).astype(int)

        DistMat = np.zeros((maxs1 - mins1) // patchSize + 1, dtype=float)

        for i in range(mins1[0], maxs1[0], patchSize):
            for j in range(mins1[1], maxs1[1], patchSize):
                for k in range(mins1[2], maxs1[2], patchSize):
                    hPs = patchSize // 2
                    print(i + hPs, j + hPs, k + hPs)
                    pos = Variable(torch.FloatTensor([[i + hPs, j + hPs, k + hPs]]), requires_grad=False)

                    DistMat[i // patchSize, j // patchSize, k // patchSize] = \
                        getVDistanceInLocation(mat11, mat12, mat22, x1, x2, pos, sigmaVis)

        return DistMat

    return ConvVDist


def getVDistanceInLocation(mat11, mat12, mat22, x1, x2, pos, sigmaVis):
    oos2_vis = 1 / sigmaVis ** 2

    kerx1 = torch.exp(- oos2_vis * torch.sum((x1 - pos) ** 2, dim=1))
    kerx2 = torch.exp(- oos2_vis * torch.sum((x2 - pos) ** 2, dim=1))

    Posker11 = kerx1[:, None] * kerx1[None, :]
    Posker12 = kerx1[:, None] * kerx2[None, :]
    Posker22 = kerx2[:, None] * kerx2[None, :]

    tot=K_loc_xy * K_or_xy * K_rad_xy
    tot[torch.where(torch.isnan(tot))]=0

    mP11=mat11 * Posker11
    mP11[torch.where(torch.isnan(mP11))]=0

    mP12=mat12 * Posker12
    mP12[torch.where(torch.isnan(mP12))]=0

    mP22=mat22 * Posker22
    mP22[torch.where(torch.isnan(mP22))]=0

    return torch.sqrt(torch.sum(mP11) - 2 * torch.sum(mP12) + torch.sum(mP22))













g2=ggt.load('/data_2to/p4/no_filapodia/4_graph_reduced.gt')
g1=ggt.load('/data_2to/p4/filapodia/4_graph_reduced_no_filopedia2.gt')

# g1=ggt.load('/data_2to/p4/no_filapodia/4_graph_cleaned.gt')
# g2=ggt.load('/data_2to/p4/filapodia/4_graph_cleaned.gt')



sigmaPos=1
sigmaRad=1


x1 = getNormalCoordinates(g1)
x2 = getNormalCoordinates(g2)

# get orientations
or1 = edgeOrientations(g1,normalize=False)
or2 = edgeOrientations(g2,normalize=False)

# get radii
rad1 = np.array(g1.edge_radii())[:, None]
rad2 = np.array(g2.edge_radii())[:, None]

# Convert everything to torch Variables
x1 = Variable(torch.from_numpy(x1).float(), requires_grad=False)
x2 = Variable(torch.from_numpy(x2).float(), requires_grad=False)
or1 = Variable(torch.from_numpy(or1).float(), requires_grad=False)
or2 = Variable(torch.from_numpy(or2).float(), requires_grad=False)
rad1 = Variable(torch.from_numpy(rad1).float(), requires_grad=False)
rad2 = Variable(torch.from_numpy(rad2).float(), requires_grad=False)
sigma1 = Variable(torch.FloatTensor(1, 1).fill_(sigmaPos), requires_grad=False)
sigma2 = Variable(torch.FloatTensor(1, 1).fill_(sigmaRad), requires_grad=False)






Dist = computeDistance(g1, g2)
CVD = ConvVarifoldDistance(sigmaPos=1, sigmaRad=1, sigmaVis=20, patchSize=20)






Dist = VarifoldDistance(sigma1, sigma2)

start = time.time()
D = float(Dist(x1, x2, or1, or2, rad1, rad2))
print("time to compute Dist on cpu : ", round(time.time() - start, 2), " seconds")

print("Dist = ", D)


DistMat = CVD(x1, x2, or1, or2, rad1, rad2)
np.save("/data_2to/p4/varDistMattest.npy", DistMat)

patchSize = 20
edgemin = 0
edgemax = 150

def col_map(x, min, max):
    res = x - min
    res /= (max - min)
    return res, 1 - res, 0.2, 1

mins1 = np.min(getNormalCoordinates(g1), axis=0).astype(int)
edge1 =getNormalCoordinates(g1)
edge1 -= mins1
edge1 /= patchSize  # halfPatchSize
edge1 = edge1.astype(int)
edge1 = np.array([DistMat[edge1[i, 0], edge1[i, 1], edge1[i, 2]] for i in range(len(edge1))])
edge1color = np.array([col_map(x, edgemin, edgemax) for x in edge1])


mins2 = np.min(getNormalCoordinates(g2), axis=0).astype(int)
edge2 = getNormalCoordinates(g2)
edge2 -= mins2
edge2 /= patchSize  # halfPatchSize
edge2 = edge2.astype(int)
edge2 = np.array([DistMat[edge2[i, 0], edge2[i, 1], edge2[i, 2]] for i in range(len(edge2))])
edge2color = np.array([col_map(x, edgemin, edgemax) for x in edge2])

p3d.plot_graph_mesh(g1, edge_colors=edge1color)
p3d.plot_graph_mesh(g2, edge_colors=edge2color)