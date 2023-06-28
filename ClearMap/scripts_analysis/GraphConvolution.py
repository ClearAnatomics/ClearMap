import numpy as np
import torch
import graph_tool.topology as gtt
from torch import nn
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader  # , Dataset
import ClearMap.Analysis.Graphs.GraphGt as ggt
import shutil
from tensorboardX import SummaryWriter
from ClearMap.Dataset import Dataset
num_epochs = 100
batch_size = 128
learning_rate = 1e-5

def init_graph(g):
    connectivity=g.edge_connectivity()
    n_vertices=g.n_vertices
    for e in g.edges:
        g.remove_edge(e)
    for i in range(n_vertices):
        for j in range(n_vertices):
            if i!=j:
                g.add_edge((i, j))
    edge_weight=np.zeros(g.n_edges)
    new_connectivivty=g.edge_connectivity()
    for n in range(g.n_edges):
        if new_connectivivty[n] in connectivity:
            edge_weight[n]=1
    g.add_edge_property('weights', edge_weight)

    return(g)


def maxpool(g, N):
    temp=torch.sum(g.edge_property('weights'), axis=1)
    factor=np.floor(float(g.n_vertices)/float(N))
    n=0
    i=0
    g_temp=g.copy()
    visited_vertices=np.zeros(g.n_vertices)

    while n <= factor:
        dist = g_temp.vertex_property_map_to_python(gtt.shortest_distance(g.base, source=None), as_array=True)
        print(dist.shape)
        filter = np.zeros(g_temp.n_vertices)
        filter[np.argsort(dist[i])[:N]] = 1
        weights=g_temp.edge_property('weights')
        v_weigths=  ggt.vertex_out_edges_iterator(g_temp, g_temp.vertices[i])
        w=weights[v_weigths]
        for n in range(g_temp.n_vertices):
            if n!=i:
                g_temp.add_edge((g_temp.n_vertices, n))
                w=np.vstack(weights, w[n])
        g_temp.remove_vertex(np.where(filter==1))
        g_temp.add_vertex()
        i=i+np.sum(filter)
        n=n+1
    return g_temp


def pooling(g, N):
   for n in range(N):
       visited_indices=[]
       coarsening_array=np.zeros(g.n_vertices)
       s = gtt.vertex_similarity(g, "jaccard")
       for i in range(g.n_vertices):
           if i not in visited_indices:
               indices=np.where(np.max(s[i].a))
               for index in indices:
                   if index!=i:
                       if index not in visited_indices:
                           visited_indices.append(i)
                           visited_indices.append(index)
                           break

               coarsening_array[i]=i
               coarsening_array[index] = i

       coarsed_graph=ggt.Graph(n_vertices=len(np.unique(coarsening_array)))
       connectivity=g.edge_connectivity()
       count=0
       for u in np.unique(coarsening_array):
           u_connect=connectivity[np.where(coarsening_array==u)]
           connected_group=np.unique(u_connect)
           connected_group=np.delete(connected_group,[np.where(coarsening_array==u)] )
           for c in connected_group:
                coarsed_graph.add_edge((count, coarsening_array[c]))
       count=count+1
   g=coarsed_graph
   return coarsed_graph


# return g_temp
class graph_convolution(nn.Module):
    def __init__(self, nin, nout):
        super(graph_convolution, self).__init__()
        self.conv_vertices = nn.Conv1d(1, 1, kernel_size=3, padding=(1), groups=nin)
        self.conv1 = nn.Conv1d(nin, nout, kernel_size=3)
        self.relu = nn.ReLU(True)

    def getShortestPath(self, g, N):
        dist = g.vertex_property_map_to_python(gtt.shortest_distance(g.base, source=None), as_array=True)
        print(dist.shape)
        filter = np.zeros((g.n_vertices, g.n_vertices))
        for i in range(g.n_vertices):
            filter[i][np.argsort(dist[i])[:N]] = 1
        return filter

    def forward(self, g, N, s_in, s_out):
        filter=self.getShortestPath(g, N)
        vector=g.vertex_property('vp')
        for i in range(g.n_vertices):
            vp=vector[filter[i, :]]
            vp=torch.mean(self.relu(self.conv_vertices(vp)))
            vector[i]=vp
        vector=self.conv1(vector)
        g.set_vertex_property_map('vp', vector)
        return g


# def upsample(g, N):
class CNNTest(nn.Module):
    def __init__(self):
        super(CNNTest, self).__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = graph_convolution(1, 32)
        self.conv2 = graph_convolution(32, 64)
        self.softmax = nn.Softmax(dim=1)
        self.fc=nn.Linear(64, 512)


    def forward(self, x):
        x=self.conv1(x)
        x=pooling(x, 4)
        x = self.conv2(x)
        x = pooling(x, 4)
        res=self.fc(x.vertex_property('vp'))
        res=self.softmax(res)
        return res

def xavier_init(m):
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_normal_(m.weight)

# def getGraphFromLaplacian(L):


class graph_maxpool(nn.Module):
    def __init__(self, nin, nout):
        super(graph_maxpool, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding=(1, 1), dilation=1)


    def forward(self, g):
        L=g.vertex_property('weights')
        L_out = self.conv1(L)
        # g_out = getGraphFromLaplacian(L_out)
        return g


def CNN(savedmodeldir, datadir, logdir, num_epochs):
    if not os.path.exists(savedmodeldir):
        os.makedirs(savedmodeldir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        if os.listdir(logdir):
            ans = input("Empty log directory " + str(logdir) + "? (Y/N)")
            if ans == "y" or ans == "Y":
                shutil.rmtree(logdir)
                print("empty directory !")

    writer = SummaryWriter(logdir)

    #####################################################

    num_iteration = 0

    print("load dataset")
    dataset = Dataset(datadir)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

    print("Initialize model")
    if torch.cuda.is_available():
        model = CNNTest().cuda()
    else:
        model = CNNTest()

    model.apply(xavier_init)
    # model.apply(xavier_init)

    criterion1 = nn.MSELoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cpt = 0

    for epoch in range(num_epochs):
        for data in dataloader:

            cpt += 1

            if torch.cuda.is_available():
                g = data.float().cuda()
            else:
                g = data.float()

            g = init_graph(g)
            g = Variable(g)
            output = model(g)  # , nonbinlatent = model(img)
            reconstruction_loss = criterion1(output, g)

            loss = reconstruction_loss
            # ===================get infos=====================
            writer.add_scalar('Train/Loss', loss, num_iteration)


            if num_iteration % 50 == 0:
                writer.add_image('Train/Input', g.data[0, 1, :, :, 20], num_iteration)
                writer.add_image('Train/Output', output.data[0, 1, :, :, 20], num_iteration)

                # ===================backward====================
                optimizer_model.zero_grad()
                loss.backward()
                #
                print('loss', loss.data)

                optimizer_model.step()


                num_iteration += 1

                if num_iteration % 400 == 0:
                    torch.save(model.state_dict(),
                               os.path.join(savedmodeldir, 'sim_autoencoder' + str(num_iteration) + '.pth'))
                # ===================log========================
            print('epoch [{}/{}]'
                  .format(epoch + 1, num_epochs))

            torch.save(model.state_dict(), os.path.join(savedmodeldir, 'sim_autoencoder.pth'))


if __name__ == "__main__":
    #main()
    CNN()