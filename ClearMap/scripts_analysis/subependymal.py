import numpy as np
#alignment
import ClearMap.Alignment.Annotation as ano
import ClearMap.Alignment.Resampling as res
import ClearMap.Analysis.Graphs.GraphGt as grp
import ClearMap.Analysis.Graphs.GraphGt_new as grpn
import ClearMap.IO.IO as io
import matplotlib


voxelisation=False
plotGraph=False
normed=False

### get read of the value 44832 in annotation file
annotation_file=io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/Andromachi_annotation_L.tif')[:, :, :246].astype('float32')
annotatin_real=io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_1-246Ld.nrrd')


largeval=np.where(annotation_file>2000)

np.unique(annotation_file[largeval])
np.unique(annotatin_real[largeval])

annotation_file[largeval]=annotatin_real[largeval]

np.unique(annotation_file[largeval])

pbarray=np.where(annotation_file==44832)
annotation_file[pbarray]=312782656.0
io.write('/home/sophie.skriabine/Documents/ClearMap_Ressources/Andromachi_annotation_L_corrected.tif', annotation_file)




annotation_file=io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/Andromachi_annotation_L_corrected.tif')
ano.set_annotation_file('/home/sophie.skriabine/Documents/ClearMap_Ressources/Andromachi_annotation_L_corrected.tif')


controls = ['142L', '158L', '162L', '164L']
Length_CPU=[]
Length_SVZ=[]
Length_C=[]

Radii_CPU=[]
Radii_SVZ=[]
Radii_C=[]

for c in controls:
    graph_reduced=grpn.load('/data_SSD_2to/whiskers_graphs/new_graphs/'+c+'/data_graph_correcteduniverse.gt')
    # ano.set_annotation_file(
    #     '/home/sophie.skriabine/Documents/ClearMap_Ressources/Andromachi_annotation_L_corrected.tif')
    labels_normal=graph_reduced.vertex_annotation()
    ano.set_annotation_file(annotation_file)
    # graph_reduced.annotate_coordinates(ano.label_points, key='order');
    def annotation(coordinates):
        label = ano.label_points(coordinates,key='order', level=4);
        return label;



    coordinates=graph_reduced.vertex_property('coordinates_atlas')
    labels=annotation(coordinates)

    graph_reduced.annotate_properties(annotation,
                                      vertex_properties={'coordinates_atlas': 'annotation'},
                                      edge_geometry_properties={'coordinates_atlas': 'annotation'});



    # graph_reduced.save('/data_SSD_2to/whiskers_graphs/new_graphs/163L/data_graph_correcteduniverse_annotated_Andromachi.gt')
    # graph_reduced=ggto.load('/data_SSD_2to/whiskers_graphs/new_graphs/163L/data_graph_correcteduniverse.gt')


    graph=graph_reduced
    # label = graph.vertex_annotation();
    #
    #
    # level=9
    # label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    # ano.find(98, key='id')
    #
    #
    # reg=(1287, 4)
    # vertex_colors = ano.convert_label(graph.vertex_annotation(), key='order', value='rgba');
    # order, level = reg
    # label=labels
    # label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    # vertex_filter = np.zeros(graph.n_vertices)
    # vf =np.logical_or(vertex_filter, label_leveled == order)
    # vertex_colors[vf]=[1.0, 0.0,0.0,1.0]
    # graph.add_vertex_property('v_col', vertex_colors)

    if plotGraph:
        gs = graph.sub_slice((slice(1,320), slice(320,330), slice(1,228)),coordinates='coordinates_atlas');
        vertex_colors=gs.vertex_property('v_col')
        p = p3d.plot_graph_mesh(gs,vertex_colors=vertex_colors, n_tube_points=3);
        # gs = graph.sub_slice((slice(1,320), slice(528-265,528-255), slice(1,228)),coordinates='coordinates_atlas');

    '''
    subependymal zone
    =================
    id                 : 98
    atlas_id           : 719
    ontology_id        : 1
    acronym            : SEZ
    name               : subependymal zone
    color_hex_triplet  : AAAAAA
    graph_order        : 1278
    st_level           : None
    hemisphere_id      : 3
    parent_structure_id: 81
    level              : 4
    order              : 1287
    rgb                : [0.66666667 0.66666667 0.66666667]
    color_order        : 1287
    
    '''
    # label = graph.vertex_annotation();
    # label=labels


    import matplotlib.pyplot as plt
    region=[(1287, 4),(582, 7), (6,6)]#subependymal zone, #caudoputamen, #cortex



    for reg in region:
        feat=[]
        vertex_filter = np.zeros(graph.n_vertices)
        order, level = reg

        if ano.find(order, key='order')['name']!='subependymal zone':
            print('not SVZ')
            ano.set_annotation_file(annotatin_real)



            coordinates = graph_reduced.vertex_property('coordinates_atlas')
            label = annotation(coordinates)

            # graph = graph_reduced
            # label = graph_reduced.vertex_annotation();
            label=labels_normal
        else:

            # labels = annotation(coordinates)
            label = labels


        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1
        vf =np.logical_or(vertex_filter, label_leveled == order)
        if ano.find(order, key='order')['name']=='subependymal zone':
            print('remove plexus choroid')
            coordinates_y=graph.vertex_property('coordinates_atlas')[:,1]
            vf=np.logical_and(vf, coordinates_y>270)
        print(ano.find(order, key='order')['name'],np.sum(vf.astype(int)))

        g=graph.sub_graph(vertex_filter=vf)

        # vertex_colors = ano.convert_label(g.vertex_annotation(), key='order', value='rgba');
        norm = matplotlib.colors.LogNorm()  # cNorm#
        radii=g.edge_property('radii')
        ef=radii>1
        g = g.sub_graph(edge_filter=ef)
        g=g.largest_component()

        radii = g.edge_property('radii')


        if plotGraph:
            edge_colors = getColorMap_from_vertex_prop(radii,norm=None, cmx=10, cmn=2.5) ## radii

            # edge_colors = getColorMap_from_vertex_prop(radii, norm=None, cmx=50, cmn=5) ## length
            g.add_edge_property('color', edge_colors)
            # gs = graph.sub_slice((slice(1, 320), slice(270, 280), slice(1, 228)), coordinates='coordinates_atlas');
            col = g.edge_property('color')
            p = p3d.plot_graph_mesh(g,edge_colors=edge_colors, n_tube_points=3);

            # g.save('/data_SSD_2to/whiskers_graphs/new_graphs/163L/data_graph_subependymal_annotated_Andromachi.gt')


        ## get branching point densities
        if ano.find(order, key='order')['name'] == 'subependymal zone':
            volume=np.sum(annotation_file[:, 270:, :]==ano.find(order, key='order')['id'])*(25*25*25)*1e-9
        else:
            volume = np.sum(annotation_file == ano.find(order, key='order')['id']) * (25 * 25 * 25) * 1e-9
        bp=g.n_vertices

        print('volume : ', volume)
        print('bp : ', bp)
        if ano.find(order, key='order')['name']=='Isocortex':
            density=6.4e3
        else:
            density=bp/volume

        print('density : ', density )

        ## get vessels length distribution
        # plt.figure()
        bins=np.arange(0,150, 2)
        hist_length, bins_length = np.histogram(g.edge_geometry_lengths(), bins=bins, normed=normed)
        # plt.hist(g.edge_geometry_lengths(), bins=bins)
        # plt.title('edge length distribution in '+ano.find(order, key='order')['name'])

        # plt.figure()
        bins = np.arange(0, 15, 2)
        hist_radii, bins_radii = np.histogram(g.edge_radii(), bins=bins, normed=normed)
        # plt.hist(g.edge_radii(), bins=bins)
        # plt.title('edge length distribution in ' + ano.find(order, key='order')['name'])


        if reg==(1287, 4):
            Length_SVZ.append(hist_length)
            Radii_SVZ.append(hist_radii)
        elif reg==(582, 7):
            Length_CPU.append(hist_length)
            Radii_CPU.append(hist_radii)
        elif reg==(6,6):
            Length_C.append(hist_length)
            Radii_C.append(hist_radii)
        # plt.figure()
        # bins = np.arange(0, 150, 2)
        # hist_length, bins_length = np.histogram(g.edge_geometry_lengths(), bins=bins, normed=normed)
        # # plt.hist(g.edge_geometry_lengths(), bins=bins)
        # plt.title('edge length distribution in ' + ano.find(order, key='order')['name'])
import pandas as pd
import seaborn as sns

## lenght
plt.figure()
Cpd=pd.DataFrame(np.array(Length_C)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Length distribution in ' + 'cortex', size='x-large')
plt.xlabel("length values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')

plt.figure()
Cpd=pd.DataFrame(np.array(Length_SVZ)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Length distribution in ' + 'subependymale area', size='x-large')
plt.xlabel("length values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')


plt.figure()
Cpd=pd.DataFrame(np.array(Length_CPU)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Length distribution in ' + 'caudoputamen', size='x-large')
plt.xlabel("length values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')


## radii
plt.figure()
Cpd=pd.DataFrame(np.array(Radii_C)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Radii distribution in ' + 'cortex', size='x-large')
plt.xlabel("Radii values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')

plt.figure()
Cpd=pd.DataFrame(np.array(Radii_SVZ)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Radii distribution in ' + 'subependymale area', size='x-large')
plt.xlabel("Radii values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')


plt.figure()
Cpd=pd.DataFrame(np.array(Radii_CPU)).melt()
sns.lineplot(x="variable", y="value", data=Cpd)#, err_style='bars'
sns.despine()
plt.title('Radii distribution in ' + 'caudoputamen', size='x-large')
plt.xlabel("Radii values", size='x-large')
plt.ylabel(" count", size='x-large')
plt.yticks(size='x-large')
plt.xticks(np.arange(0,bins.shape[0], bins.shape[0]/10), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
plt.tight_layout()
plt.yscale('linear')



if voxelisation:
    ### voxelization of the subependymal region
    work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
    radius=5
    template_shape=(320,528,228)
    vox_shape = (320, 528, 228, len(controls))
    vox_ori_control_rad = np.zeros(vox_shape)
    feat='radii'#length

    for c in controls:
        graph=ggt.load(work_dir+'/'+c+'/data_graph_correcteduniverse.gt')

        graph.annotate_properties(annotation,
                                          vertex_properties={'coordinates_atlas': 'annotation'},
                                          edge_geometry_properties={'coordinates_atlas': 'annotation'});

        vertex_filter = np.zeros(graph.n_vertices)
        order, level = reg

        if ano.find(order, key='order')['name'] != 'subependymal zone':
            print('not SVZ')
            ano.set_annotation_file(annotatin_real)
            coordinates = graph_reduced.vertex_property('coordinates_atlas')
            label = annotation(coordinates)
            print('take real annotation')
            label = graph.vertex_annotation();
        else:
            label = labels

        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vf = np.logical_or(vertex_filter, label_leveled == order)
        if ano.find(order, key='order')['name'] == 'subependymal zone':
            # remove plexus choroid
            coordinates_y = graph.vertex_property('coordinates_atlas')[:, 1]
            vf = np.logical_and(vf, coordinates_y > 270)
        print(ano.find(order, key='order')['name'], np.sum(vf.astype(int)))

        g = graph.sub_graph(vertex_filter=vf)

        # vertex_colors = ano.convert_label(g.vertex_annotation(), key='order', value='rgba');
        norm = matplotlib.colors.LogNorm()  # cNorm#
        radii = g.edge_property('radii')
        ef = radii > 1
        g = g.sub_graph(edge_filter=ef)
        g = g.largest_component()

        radii = g.edge_property('radii')
        # coord=g.edge_property('coordinates')
        connectivity = graph.edge_connectivity()
        coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
        edges_centers = np.array(
            [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])


        vox_data = np.concatenate((edges_centers, np.expand_dims(flow_reg, axis=1)), axis=1)
        v = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=vox_data[:, 3], radius=(radius, radius, radius),
                         method='sphere');
        w = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),
                         method='sphere');
        vox_ori_control_rad[:, :, :, i] = v.array / w.array

    io.write('/home/sophie.skriabine/Pictures/subependymal' + '/' + 'vox_'+feat + str(radius) + '.tif', vox_ori_control_rad.astype('float32'))
    vox_ori_control_rad_avg = np.mean(vox_ori_control_rad, axis=3)
    io.write('/home/sophie.skriabine/Pictures/subependymal' + '/' + 'vox_'+feat + str(radius) + '.tif', vox_ori_control_rad_avg.astype('float32'))


