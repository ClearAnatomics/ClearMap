import matplotlib.pyplot as plt
ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
               annotation_file = annotation_file)

ps=7.3

work_dir='/data_SSD_2to/191122Otof'
controls=['2R','3R','5R', '8R']
control='2R'

graph = ggt.load(
    work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)
print(graph)
with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

pressure = np.asarray(sampledict['pressure'][0])
flow=np.asarray(sampledict['flow'][0])

angle, graph =  GeneralizedRadPlanorientation(graph, control, 4.5, controls, mode=mode, average=average)
graph.add_edge_property('angle', angle)

label=graph.vertex_annotation();
region=[(54, 9), (13, 7)]

for reg in region:
    vertex_filter = np.zeros(graph.n_vertices)
    order, level = reg
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = np.logical_or(vertex_filter, label_leveled == order)


    edge_filter=from_v_prop2_eprop(graph, vertex_filter)
    flow_reg = flow[np.asarray(edge_filter==1).nonzero()[0]]
    angle_reg=angle[np.asarray(edge_filter==1).nonzero()[0]]

    e = 1 - np.exp(-(ps / abs(flow_reg)))
    e = e[~np.isnan(e)]
    a=angle_reg[~np.isnan(e)]

    plt.figure()
    plt.scatter(a, flow_reg ,color=ano.find(order, key='order')['rgb'], alpha=0.005)
    plt.title(ano.find(order, key='order')['name'])

