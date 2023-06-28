
# from tqdm import tqdm
from ClearMap.GraphEmbedding import *
from ClearMap.DiffusionPenetratingArteriesCortex import diffusion_through_penetrating_arteries,get_penetration_veins_dustance_surface,get_penetrating_veins_labels
# from ClearMap.GraphMP import *
# from ClearMap.GraphCorrection import *

from ClearMap.Gt2iG import computeFlowFranca, gt2ig
from ClearMap.SBMclusters import modularity_measure
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']

mutants = ['1', '2', '4', '7', '8', '9']
work_dir = '/data_2to/DBA2J'




try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

# with open('/data_SSD_2to/191122Otof/reg_list_full.p', 'rb') as fp:
#   reg_list = pickle.load(fp)
#
# with open('/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
#   atlas_list = pickle.load(fp)

with open('/data_SSD_2to/191122Otof/id_reg_list_full.p', 'rb') as fp:
	  reg_list = pickle.load(fp)

with open('/data_SSD_2to/181002_4/atlas_id_volume_list.p', 'rb') as fp:
	atlas_list = pickle.load(fp)


def remove_surface(graph, width):
	distance_from_suface = graph.vertex_property('distance_to_surface')
	ef=distance_from_suface>width
	g=graph.sub_graph(vertex_filter=ef)
	return g


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    print(ellipse)
    return ellipse #ax.add_patch(ellipse)




def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vx, vy)

    # Width and height of ellipse to draw
    width,height = 2 * nstd * np.sqrt(eigvals)
    print(centre,width,height )
    return Ellipse(xy=centre, width=width, height=height,angle=np.degrees(theta), **kwargs)

### FEATURES EXTRACTION
# #
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls, mutants]

# #
work_dir='/data_SSD_2to/whiskers_graphs/fluoxetine'
mutants=['1','2','3', '4', '6', '18']
controls=['21','22', '23']
states=[controls, mutants]

work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']
states=[controls, mutants]

work_dir = '/data_SSD_1to/10weeks'
mutants = ['1L', '2L', '3L', '4L']
controls=['6L', '8L', '9L']
states=[controls, mutants]

mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'
states=[controls, mutants]

work_dir = '/data_SSD_1to/otof1month'
# mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']\
controls=['7', '9', '11']
mutants=[ '14', '17', '18']
states=[controls, mutants]

work_dir = '/data_2to/fluoxetine'
# mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']\
controls=['1', '2', '3', '4', '5']
mutants=[ '7', '8',]
states=[controls, mutants]

work_dir = '/data_SSD_2to/fluoxetine2'
controls=['1', '2', '3', '4','5']
mutants=[ '7', '8', '9', '10', '11']
states=[controls, mutants]


work_dir = '/data_SSD_2to/earlyDep'
controls=['4', '7', '10','12', '15']
mutants=[ '3', '6', '9', '11', '16']
states=[controls, mutants]


work_dir='/data_2to/earlyDep_ipsi'
controls=['4', '7', '10', '15']
mutants=[ '3', '6', '11', '16']
states=[controls, mutants]



work_dir='/data_2to/otof1M'
controls=[ '1w', '3w', '5w', '6w', '7w']
mutants=['1k', '2k', '3k', '4k']
states=[controls, mutants]



work_dir='/data_2to/otof3M/new_vasc'
# controls=['1w', '2w', '4w', '5w','6w']
# mutants=[ '1k','3k','4k', '5k', '6k']
controls=['2w', '4w', '5w','6w']
mutants=[ '3k', '5k', '6k']
states=[controls, mutants]


work_dir='/data_2to/whiskers5M/R'
mutants=['433', '457', '458']#456 not annotated ?
controls=['467', '468', '469']
states=[controls, mutants]
# mutants = ['1', '2', '4', '7', '8', '9']
# work_dir = '/data_2to/DBA2J'


work_dir='/data_SSD_2to/211019_otof_10m'
mutants=['1k', '2k','3k', '6k']#456 not annotated ?
controls=['7w', '9w', '10w', '12w', '13w']
states=[controls, mutants]


work_dir='/data_SSD_2to/220503_p14_otof'
controls=['1w', '2w', '4w']
mutants=['5k', '6k', '7k', '8k']
states=[controls, mutants]

work_dir='/data_2to/gestantes/'
controls=['1v', '2v', '3v', '4v', '5v']
mutants=['1p', '2p', '3p', '4p', '5p']
states=[controls, mutants]


side='L'
work_dir='/data_SSD_2to/otofRECUP/'
controls=['17'+side, '20'+side,'22'+side]
mutants=['18'+side, '19'+side,'21'+side]
states=[controls, mutants]






# states=[mutants]
### GRAPH CORRECTION
# brains=['138L', '141L', '142L', '158L', '163L', '162L', '164L','165L']
# brains = ['36']#['21', '22', '23']#, ['1', '2', '3', '4', '6', '18', '26']
# for c in brains:
#     graph=ggt.load(work_dir+'/'+c+'/data_graph.gt')
#     giso, T = graphCorrection(graph, work_dir+'/'+c, region_list[0], save=True)
#
#
# states=[controls, mutants]

ls=['l1', 'l2/3','l4','l5','l6a','l6b']


mode='clusters'#'layers'
ext_step=7
compute_distance=False
compute_loops=False
compute_path=False
# basis = CreateSimpleBasis(3, 7)
sub_region=False
condition='isocortex'#'isocortex'#'isocortex'#'barrel_region'#'Auditory_regions'
feature='vessels'#'vessels'#art_raw_signal



if condition == 'Aud_p':
    region_list = [(142, 8)]  # auditory
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'Aud':
    region_list = [(127, 7)]  # barrels
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
if condition == 'Aud_po':
    region_list = [(149, 8)]  # auditory
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
if condition == 'Aud_d':
    region_list = [(128, 8)]  # auditory
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
if condition == 'Aud_v':
    region_list = [(156, 8)]  # auditory
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'Ssp':
    region_list = [(40, 8)]  # barrels
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'barrels':
    region_list = [(54, 9)]  # barrels
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'nose':
    region_list = [(47, 9)]  # barrels
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'mouth':
    region_list = [(75, 9)]  # barrels
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        n = ano.find_name(r, key='order')
        if R in n:
            for se in reg_list[r]:
                n = ano.find_name(se, key='order')
                print(n)
                regions.append(n)
elif condition == 'Auditory_regions':
    region_list = [[(142, 8), (149, 8), (128, 8), (156, 8)]]
    sub_region = True
elif condition == 'barrel_region':
    regions = [[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels
    sub_region = True
elif condition == 'l2 barrels':
    regions = [(56, 10), (49, 10), (77, 10)]  # barrels
elif condition == 'l4 barrels':
    regions = [(58, 10), (51, 10), (79, 10)]  # barrels
elif condition == 'isocortex':
    region_list = [(315, 6)]  # isocortex
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        l = ano.find(r, key='id')['level']
        regions.append([(r, l)])

# new_id_reg_list={}
# for r in reg_list.keys():
# 	i= ano.find(r, key='order')['id']
# 	l = ano.find(r, key='order')['level']
# 	new_id_reg_list[i]=[ano.find(ord, key='order')['id'] for  ord in reg_list[r]]
#
# new_atlas_reg_list={}
# for r in atlas_list.keys():
# 	i= ano.find(r, key='order')['id']
# 	l = ano.find(r, key='order')['level']
# 	new_atlas_reg_list[i]= atlas_list[r]
#
#
# try:
# 	import cPickle as pickle
# except ImportError:  # python 3.x
# 	import pickle
#
# with open('/data_SSD_2to/191122Otof/id_reg_list_full.p', 'wb') as fp:
# 	pickle.dump(new_id_reg_list, fp)
#
# with open('/data_SSD_2to/181002_4/atlas_id_volume_list.p', 'wb') as fp:
# 	pickle.dump(new_atlas_reg_list, fp)

# regions = [[(54, 9)], [(25, 8)]]

def distance(coordinates):
	c = np.asarray(np.round(coordinates), dtype=int);
	c[c < 0] = 0;
	x = c[:, 0];
	y = c[:, 1];
	z = c[:, 2];
	x[x >= distance_file_shape[0]] = distance_file_shape[0] - 1;
	y[y >= distance_file_shape[1]] = distance_file_shape[1] - 1;
	z[z >= distance_file_shape[2]] = distance_file_shape[2] - 1;
	d = distance_file[x, y, z];
	return d;

def annotation(coordinates):
	label = ano.label_points(coordinates, key='id');
	return label;


distance_file ='/data_2to/gestantes/279 - ABA_25um_distance_to_surface__-1_2_3__slice_None.tif'
annotation_file='/data_2to/gestantes/ABA_25um_annotation__-1_2_3__slice_None_None_None__slice_None_None_None__slice_0_240_None__.tif'
distance_file = io.read(distance_file)
distance_file_shape = distance_file.shape;

ano.set_annotation_file(annotation_file)

for control in ['5v']:
	if control in ['5p', '4v', '5v']:
		distance_file ='/data_2to/gestantes/279 - ABA_25um_distance_to_surface__1_-2_3__slice_None.tif'
		annotation_file='/data_2to/gestantes/ABA_25um_annotation__1_-2_3__slice_None_None_None__slice_None_None_None__slice_0_240_None__.tif'
		distance_file = io.read(distance_file)
		ano.set_annotation_file(annotation_file)

	graph=ggt.load(work_dir+'/'+control+'/'+control+'_graph_annotated.gt')
	# try:
	# 	graph=remove_surface(graph, 2)
	# except:
	print('no distance to surface attribute')
	graph.transform_properties(distance,
							   vertex_properties={'coordinates_atlas': 'distance_to_surface'},
							   edge_geometry_properties={'coordinates_atlas': 'distance_to_surface'});

	distance_to_surface = graph.edge_geometry('distance_to_surface', as_list=True);
	distance_to_surface__edge = np.array([np.min(d) for d in distance_to_surface])
	graph.define_edge_property('distance_to_surface', distance_to_surface__edge)
	graph.annotate_properties(annotation,
							  vertex_properties={'coordinates_atlas': 'annotation'},
							  edge_geometry_properties={'coordinates_atlas': 'annotation'});
	graph=remove_surface(graph, 2)
	graph=graph.largest_component()
	print(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')
	graph.save(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')


compute_SBM=False
compute_flow=False
compute_prop_ori=True
orientation_method = 'localNormal'#'localNormal'#flowInterpolation
limit_angle=40
mode='bigvessels'
average=False


BP_controls=[]
BP_mutants=[]
BP_recup=[]

EP_controls=[]
EP_mutants=[]
EP_recup=[]

ORI_controls=[]
ORI_mutants=[]
ORI_recup=[]


ARTBP_controls=[]
ARTBP_mutants=[]
ARTBP_recup=[]

VDIST_controls=[]
VDIST_mutants=[]
VDIST_recup=[]


EDIST_controls=[]
EDIST_mutants=[]
EDIST_recup=[]

if compute_SBM:
	SBMQ_controls=[]
	SBMNB_controls=[]

	SBMQ_mutants=[]
	SBMNB_mutants=[]

if compute_flow:
	FLOW_controls=[]
	FLOW_mutants=[]

if compute_prop_ori:
	PROP_ORI_RAD_controls=[]
	PROP_ORI_RAD_mutants = []
	PROP_ORI_RAD_recup = []


import pickle

# controls=['3R']
# states=[controls]
for state in states:
	for a, control in enumerate(state):
		print(control)
		length_short_path_control = []
		ep = []
		bp = []
		ori=[]
		q=[]
		nb=[]
		flow=[]
		art_bp=[]
		prop_ori=[]

		bp_dist_2_surface = []
		try:
			graph = ggt.load(work_dir + '/' + control + '/' +control+ '_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
		except FileNotFoundError:
			graph = ggt.load(work_dir + '/' + control + '/' +'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#

		degrees = graph.vertex_degrees()
		vf = np.logical_and(degrees > 1, degrees <= 4)
		graph = graph.sub_graph(vertex_filter=vf)


		# diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
		# graph.add_vertex_property('overlap', diff)

		try:
			artery=graph.vertex_property('artery')
			vein=graph.vertex_property('vein')
			# artery=from_e_prop2_vprop(graph, 'artery')
			# vein=from_e_prop2_vprop(graph, 'vein')
		except:
			try:
				artery=from_e_prop2_vprop(graph , 'artery')
				vein=from_e_prop2_vprop(graph , 'vein')
			except:
				print('no artery vertex properties')
				artery=np.logical_and(graph.vertex_radii()>=4.8,graph.vertex_radii()<=8)#4
				vein=graph.vertex_radii()>=8
				graph.add_vertex_property('artery', artery)
				graph.add_vertex_property('vein', vein)
				artery=from_v_prop2_eprop(graph, artery)
				graph.add_edge_property('artery', artery)
				vein=from_v_prop2_eprop(graph, vein)
				graph.add_edge_property('vein', vein)

		if compute_flow:
			print("compute flow...")
			try:
				with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
					sampledict = pickle.load(fp)

				f = np.asarray(sampledict['flow'][0])
				v = np.asarray(sampledict['v'][0])
				graph.add_edge_property('flow', f)
				graph.add_edge_property('veloc', v)
				pressure = np.asarray(sampledict['pressure'][0])
				graph.add_vertex_property('pressure', pressure)
			except:
				f, v=computeFlowFranca(work_dir, graph,control)
				with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
					sampledict = pickle.load(fp)
				p = np.asarray(sampledict['pressure'][0])
				graph.add_edge_property('flow', f)
				graph.add_edge_property('veloc', v)
				graph.add_vertex_property('pressure', p)
			print('done')

		if orientation_method == 'flowInterpolation':
			if average==True:
				angle, graph =  GeneralizedRadPlanorientation(graph, control, 4.5, controls, mode=mode, average=average)
				graph.add_edge_property('angle', angle)

			# with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
			# 	sampledict = pickle.load(fp)
			#
			# pressure = np.asarray(sampledict['pressure'][0])
			# graph.add_vertex_property('pressure', pressure)



		for region_list in regions:
			

			vertex_filter = np.zeros(graph.n_vertices)
			for i, rl in enumerate(region_list):
				id, level = region_list[i]
				print(level, id, ano.find(id, key='id')['name'])
				label = graph.vertex_annotation();
				label[label>=100000]=0 # filters out weird regions
				label_leveled = ano.convert_label(label, key='id', value='id', level=level)
				vertex_filter[label_leveled == id] = 1;
			gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
			print(gss4_t)
			gss4=gss4_t.copy()
			nv=gss4.n_vertices
			ne=gss4.n_edges
			bp_dist_2_surface.append(gss4.vertex_property('distance_to_surface'))
			bp.append(gss4.vertex_property('distance_to_surface'))
			# ep.append(gss4.edge_property('distance_to_surface'))
			print('compute orientation ...')
			if orientation_method == 'localNormal':
				try:
					r, p, n, l = getRadPlanOrienttaion(gss4, gss4_t,local_normal=True, calc_art=False) # ,, , calc_art=True)

					if not compute_prop_ori:
						r1 = r[~np.isnan(r)]
						p = p[~np.isnan(r)]
					elif compute_prop_ori:
						r1 = r[~np.isnan(r)]
						p = p[~np.isnan(r)]
						dist = gss4.edge_property('distance_to_surface')[~np.isnan(r)]
						ep.append(dist)

						# radiality = (r / (r + p)) > 0.5
						# planarity = (p / (r + p)) > 0.6
						# neutral = np.logical_not(np.logical_or(radiality, planarity))
						angle = np.array([math.acos(r1[i]) for i in range(r1.shape[0])]) * 180 / pi

						radiality = angle <  limit_angle#40
						planarity = angle >  90-limit_angle#60
						neutral = np.logical_not(np.logical_or(radiality, planarity))

						ori_prop = np.concatenate((np.expand_dims(dist, axis=1),np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
							(np.expand_dims(neutral, axis=1),np.expand_dims(planarity, axis=1)), axis=1)), axis=1)), axis=1)
						prop_ori.append(ori_prop)

						rad = angle <  limit_angle  # 40
						plan = angle >  90 - limit_angle  # 60
				except:
					print('indexError')
					r,p,n,l=np.zeros(gss4.n_edges),np.zeros(gss4.n_edges),np.zeros(gss4.n_edges),np.zeros(gss4.n_edges)
					dist = np.zeros(gss4.n_edges)
					angle=np.zeros(gss4.n_edges)
					rad = np.zeros(gss4.n_edges)
					plan = np.zeros(gss4.n_edges)

			elif orientation_method == 'flowInterpolation':
				try:
					if average:
						angle=gss4.edge_property('angle')
					else:
						angle, graph =  GeneralizedRadPlanorientation(graph, g, 4.5, controls, mode=mode, average=average)
					# angle = GeneralizedRadPlanorientation(gss4, control, mode='bigvessels')
						# angle = GeneralizedRadPlanorientation(gss4, control, mode=mode, average=average)

					dist = gss4.edge_property('distance_to_surface')
					ep.append(dist)

					radiality=angle < limit_angle#40
					planarity=angle > 90-limit_angle#60


					neutral = np.logical_not(np.logical_or(radiality, planarity))

					ori_prop = np.concatenate((np.expand_dims(dist, axis=1), np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
						(np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)), axis=1)
					prop_ori.append(ori_prop)

					rad = angle < limit_angle  # 40
					plan = angle > 90 - limit_angle  # 60

				except:
					print('indexError - pb in generalized ORI')
					r,p,n,l=np.zeros(gss4.n_edges),np.zeros(gss4.n_edges),np.zeros(gss4.n_edges),np.zeros(gss4.n_edges)
					dist = np.zeros(gss4.n_edges)
					angle=np.zeros(gss4.n_edges)
					rad = np.zeros(gss4.n_edges)
					plan = np.zeros(gss4.n_edges)

			# rad = (r / (r + p)) > 0.5
			# plan = np.sum((p / (r + p)) > 0.6)
			# angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

			ori.append(rad)
			print('done')
			artery = from_e_prop2_vprop(gss4_t, 'artery')
			# vertex_filter=np.logical_and(artery,gss4_t.vertex_property('artery_binary')>0)#np.logical_and()
			# art_g = gss4_t.sub_graph(vertex_filter=vertex_filter)
			# dist_art = art_g.vertex_property('distance_to_surface')
			# art_bp.append(dist_art)

			if compute_flow:
				f=gss4.edge_property('flow')
				v=gss4.edge_property('veloc')

				flow.append(f)
			
			Nb_vertices = gss4.n_vertices
			rad = gss4.vertex_property('radii')
			connect = gss4.edge_connectivity()
			leng = gss4.edge_property('length')
			param = [1 / (1 + abs(1 - (rad[connect[i][0]] / rad[connect[i][1]]))) for i in np.arange(gss4.n_edges)]
			gss4.add_edge_property('param', np.array(param).astype(float))
			gss4.add_edge_property('invL', 1 / leng)

			if compute_SBM:
				print('sbm ... ')
				Qs=[]
				N=[]
				gss4.add_edge_property('rp', p / (r + p))
				g = gss4.base

				for i in range(5):
					state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],rec_types=["real-exponential", "real-exponential", "real-exponential"]))
					modules=state.get_blocks().a
					gss4.add_vertex_property('blocks',modules)
					try:
						Q, Qbis = modularity_measure(modules, gss4, 'blocks')
					except:
						Q=0
					s = np.unique(modules).shape[0]
					Qs.append(Q)
					N.append(s)
				q.append(np.mean(np.array(Qs)))
				nb.append(np.mean(np.array(N)))
				print('done')
		if state==controls:
			BP_controls.append(bp)
			ORI_controls.append(ori)
			PROP_ORI_RAD_controls.append(prop_ori)
			EP_controls.append(ep)
			# ARTBP_controls.append(art_bp)
			if compute_SBM:
				SBMQ_controls.append(q)
				SBMNB_controls.append(nb)
			if compute_flow:
				FLOW_controls.append(flow)
		elif state==mutants:
			BP_mutants.append(bp)
			ORI_mutants.append(ori)
			PROP_ORI_RAD_mutants.append(prop_ori)
			EP_mutants.append(ep)
			# ARTBP_mutants.append(art_bp)
			if compute_SBM:
				SBMQ_mutants.append(q)
				SBMNB_mutants.append(nb)
			if compute_flow:
				FLOW_mutants.append(flow)

		elif state==recup:
			BP_recup.append(bp)
			ORI_recup.append(ori)
			PROP_ORI_RAD_recup.append(prop_ori)
			EP_recup.append(ep)
			# ARTBP_mutants.append(art_bp)
			if compute_SBM:
				SBMQ_recup.append(q)
				SBMNB_recup.append(nb)
			if compute_flow:
				FLOW_recup.append(flow)

		if orientation_method == 'flowInterpolation':
			np.save(work_dir + '/' + 'ORI_fi_bv' + condition +  '_'+control+'.npy', ori)
			np.save(work_dir + '/' + 'PROP_ORI_fi_bv' + condition + '_' + control + '.npy', prop_ori)
		elif orientation_method=='localNormal':
			np.save(work_dir + '/' + 'ORI_ln_test' + condition +  '_'+control+'.npy', ori)
			np.save(work_dir + '/' + 'PROP_ORI_ln_test' + condition + '_' + control + '.npy', prop_ori)

		# np.save(work_dir + '/' + 'ARTBP_' + condition +  '_'+control+'.npy', art_bp)
		if compute_SBM:
			np.save(work_dir + '/' + 'SBMQ_' + condition +  '_'+control+'.npy', q)
			np.save(work_dir + '/' + 'SBMNB_' + condition +  '_'+control+'.npy', nb)
		if compute_flow:
			np.save(work_dir + '/' + 'FLOW_' + condition +  '_'+control+'.npy', flow)
		np.save(work_dir + '/' + 'BP_' + condition +  '_'+control+'.npy', bp)
		np.save(work_dir + '/' + 'EP_' + condition +  '_'+control+'.npy', ep)



### GRAPH PLOTTING
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
# from ClearMap.Wasserstein import *
states=[controls, mutants]




work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']
states=[controls, mutants]




work_dir = '/data_SSD_1to/10weeks'
controls = ['1L', '2L', '3L', '4L']
mutants=['6L', '8L', '9L']
states=[controls, mutants]



work_dir = '/data_SSD_1to/otof1month'
# mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']\
controls=['7', '9', '11']
mutants=[ '14', '17', '18']
states=[controls, mutants]



work_dir = '/data_SSD_2to/fluoxetine2'
controls=['1', '2', '3', '4','5']
mutants=[ '7', '8', '9', '10', '11']
states=[controls, mutants]


work_dir = '/data_SSD_2to/earlyDep'
controls=['4', '7', '10','12', '15']
mutants=[ '3', '6', '11', '16']
states=[controls, mutants]



work_dir='/data_2to/earlyDep_ipsi'
controls=['4', '7', '10', '15']
mutants=[ '3', '6', '11', '16']
states=[controls, mutants]




control='controls'
feature='vessels'
nb_reg=48
N = 5
thresh=2
bin = 10
bin2 = 10 



ps=7.3

# features = []
# nb_reg=2
# brains=controls
# for control in brains:
# 	features_brains = []
# 	for reg in range(nb_reg):
# 		vess_rad_control_fi_bv=np.load(work_dir + '/' + 'ORI_fi_bv' + condition +  '_'+control+'.npy',allow_pickle=True)
# 		prop_ori_control_fi_bv=np.load(work_dir + '/' + 'PROP_ORI_fi_bv' + condition + '_' + control + '.npy',allow_pickle=True)
#
# 		vess_rad_control_fi_av=np.load(work_dir + '/' + 'ORI_fi_av' + condition +  '_'+control+'.npy',allow_pickle=True)
# 		prop_ori_control_fi_av=np.load(work_dir + '/' + 'PROP_ORI_fi_av' + condition + '_' + control + '.npy',allow_pickle=True)
#
# 		vess_rad_control_ln_test=np.load(work_dir + '/' + 'ORI_ln_test' + condition +  '_'+control+'.npy',allow_pickle=True)
# 		prop_ori_control_ln_test=np.load(work_dir + '/' + 'PROP_ORI_ln_test' + condition + '_' + control + '.npy',allow_pickle=True)
#
# 		dist=prop_ori_control_fi_bv[reg][:,0]
# 		ori_rad=prop_ori_control_fi_bv[reg][:,1]
# 		ori_neutral = prop_ori_control_fi_bv[reg][:,2]
# 		ori_plan = prop_ori_control_fi_bv[reg][:,3]
# 		histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
# 		histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
# 		histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
# 		R_fi_bv= histrad / (histrad + histneut + histplan)
# 		N_fi_bv= histneut / (histrad + histneut + histplan)
# 		P_fi_bv = histplan / (histrad + histneut + histplan)
#
# 		dist=prop_ori_control_fi_av[reg][:,0]
# 		ori_rad=prop_ori_control_fi_av[reg][:,1]
# 		ori_neutral = prop_ori_control_fi_av[reg][:,2]
# 		ori_plan = prop_ori_control_fi_av[reg][:,3]
# 		histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
# 		histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
# 		histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
# 		R_fi_av= histrad / (histrad + histneut + histplan)
# 		N_fi_av= histneut / (histrad + histneut + histplan)
# 		P_fi_av = histplan / (histrad + histneut + histplan)
#
# 		dist=prop_ori_control_ln_test[reg][:,0]
# 		ori_rad=prop_ori_control_ln_test[reg][:,1]
# 		ori_neutral = prop_ori_control_ln_test[reg][:,2]
# 		ori_plan = prop_ori_control_ln_test[reg][:,3]
# 		histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
# 		histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
# 		histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
# 		R_ln_test= histrad / (histrad + histneut + histplan)
# 		N_ln_test= histneut / (histrad + histneut + histplan)
# 		P_ln_test = histplan / (histrad + histneut + histplan)
#
# 		features_brains.append([R_fi_bv, R_fi_av, R_ln_test])
#
# 	features.append(features_brains)
#
# shape = (len(brains), len(features_brains), len(features_brains[0]), bin)
# F = np.zeros(shape).astype(float)
# F=F.astype(float)
# for i in range(shape[0]):
# 	for j in range(shape[1]):
# 		for k in range(shape[2]):
# 			F[i, j, k, :] = features[i][j][k].astype(float)
# # features=np.array(features)
# features=np.nan_to_num(F)
# features_avg=np.mean(features, axis=0)
# features_avg_c=features_avg
# features_c=features
# import seaborn as sns
# from sklearn.preprocessing import normalize
# import pandas as pd
# norm=False
# for r in range(features.shape[1]):
# 		plt.figure()
# 		sns.set_style(style='white')
#
# 		norm = False
# 		f = 0#3
# 		if norm:
# 			Cpd_c0 = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
#
# 		else:
# 			Cpd_c0 = pd.DataFrame(features_c[:, r, f]).melt()
#
#
# 		f =  1
# 		norm = False
# 		if norm:
# 			Cpd_c1 = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
# 		else:
# 			Cpd_c1 = pd.DataFrame(features_c[:, r, f]).melt()
#
# 		f =  2
# 		norm = False
# 		if norm:
# 			Cpd_c2 = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
# 		else:
# 			Cpd_c2 = pd.DataFrame(features_c[:, r, f]).melt()
#
#
# 		sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c0, color=colors_m[1], linewidth=2.5)
# 		sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c1, color=colors_c[0], linewidth=2.5)
# 		sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c2, color=colors_c[2], linewidth=2.5)
# 		plt.legend(['fi_bv', 'fi_av', 'ln'])#(['prop rad', 'prop plan'])
# 		plt.yticks(size='x-large')
# 		plt.ylabel('PROP_ORI', size='x-large')
# 		plt.tight_layout()
#
#
#






allow_pickle=True
compute_flow=False
reg_lis=True
for brains in states:
	normed = False
	regis = []
	features = []  # np.array((len(brains),nb_reg))
	first = True

	grouped = True
	aud = True
	vis = True
	rsp = True
	nose = True
	trunk = True

	reg_lis = True

	for control in brains:
		try:
			SBM_modules=np.load(work_dir + '/' + 'SBMNB_' + condition +  '_'+control+'.npy')
			SBM_modularity=np.load(work_dir + '/' + 'SBMQ_' + condition +  '_'+control+'.npy')


		except:
			print('no sbm were computed')
			SBM_modules=np.zeros(4)
			SBM_modularity=np.zeros(4)

		if compute_flow:
			flow_simulation = np.load(work_dir + '/' + 'FLOW_' + condition + '_' + control + '.npy', allow_pickle=True)

		# vess_rad_control=np.load(work_dir + '/' + feature + 'control_rad_ori' + condition + '_' + control + '.npy',allow_pickle=True)[0]
		# vess_rad_control=np.load(work_dir + '/' + 'ORI_3' + condition +  '_'+control+'.npy',allow_pickle=True)
		# prop_ori_control=np.load(work_dir + '/' + 'PROP_ORI_' + condition + '_' + control + '.npy',allow_pickle=True)
		vess_rad_control=np.load(work_dir + '/' + 'ORI_ln_test' + condition +  '_'+control+'.npy',allow_pickle=True)
		prop_ori_control=np.load(work_dir + '/' + 'PROP_ORI_ln_test' + condition + '_' + control + '.npy',allow_pickle=True)
		bp_dist_2_surface_control=np.load(work_dir + '/' + 'BP_' + condition +  '_'+control+'.npy',allow_pickle=True)
		# art_ep_dist_2_surface_control = np.load(work_dir + '/' + 'ARTBP_' + condition + '_' + control + '.npy',allow_pickle=True)
		ve_ep_dist_2_surface_control = np.load(work_dir + '/' + 'EP_' + condition + '_' + control + '.npy', allow_pickle=True)


		features_brains = []
		for reg in range(nb_reg):
			# depth = []
			id, level = regions[reg][0]
			n = ano.find(id, key='id')['acronym']
			# id=ano.find(order, key='order')['id']
			print(id,n,(n != 'NoL' or id==182305712))
			if (n != 'NoL' or id==182305712):
				if '6' not in n:
					if reg_lis:
						print(regions[reg])
						regis.append(regions[reg])
					try:
						if compute_flow:
							flow_sim=np.array(flow_simulation[reg])
							e = 1 - np.exp(-(ps / abs(flow_sim)))
							e = e[~np.isnan(e)]
						try:
							sbm_nb=np.array(SBM_modules[reg])
							sbm_q=np.array(SBM_modularity[reg])
						except:
							sbm_nb=np.zeros(10)
							sbm_q=np.zeros(10)
						bp_dist=np.array(bp_dist_2_surface_control[reg])##
						# art_ep=np.array(art_ep_dist_2_surface_control[reg])##
						ve_ep=np.array(ve_ep_dist_2_surface_control[reg])##
						ori=np.array(vess_rad_control[reg])
						# radial_ori=ori[:int(len(ori)/2)]
						# radial_depth=ori[int(len(ori)/2):]
						radial_depth=ve_ep[ori>0.6]
						if compute_flow:
							hist_flow, bins_flow = np.histogram(e, bins=bin2, normed=normed)
						# hist_art_ep, bins_art_ep = np.histogram(art_ep, bins=bin, normed=normed)
						hist_ve_ep, bins_ve_ep = np.histogram(ve_ep, bins=bin, normed=normed)
						hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist>thresh], bins=bin, normed=normed)
						#, np.sum(np.mean(H, axis=1)))
						hist_ori, bins_ori_dist = np.histogram(radial_depth, bins=bin, normed=normed)

						dist=prop_ori_control[reg][:,0]
						ori_rad=prop_ori_control[reg][:,1]
						ori_neutral = prop_ori_control[reg][:,2]
						ori_plan = prop_ori_control[reg][:,3]
						histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
						histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
						histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
						R = histrad / (histrad + histneut + histplan)
						N = histneut / (histrad + histneut + histplan)
						P = histplan / (histrad + histneut + histplan)

						if compute_flow:
							print('features with flow sim')
							# features_brains.append([hist_art_ep,hist_ve_ep,hist_bp_dist,hist_ori,sbm_q,sbm_nb, hist_flow, P, N, R])#len(shortest_paths_control)
							features_brains.append([np.zeros(10), hist_ve_ep,hist_bp_dist,hist_ori,sbm_q,sbm_nb, hist_flow, P, N, R])#len(shortest_paths_control)
						else:
							print('features without flow sim')
							# features_brains.append([hist_art_ep, hist_ve_ep, hist_bp_dist, hist_ori, sbm_q, sbm_nb, P, N,R])  # len(shortest_paths_control)
							features_brains.append([np.zeros(10), hist_ve_ep, hist_bp_dist, hist_ori, sbm_q, sbm_nb, P, N,R])  # len(shortest_paths_control)
						print(reg, 'works!')
					except:
						print(reg,'no data')
						features_brains.append([np.zeros(10),np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)])
		
		features.append(features_brains)
		reg_lis=False

	shape = (len(brains), len(features_brains), len(features_brains[0]), bin)
	F = np.zeros(shape).astype(float)
	F=F.astype(float)
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				F[i, j, k, :] = features[i][j][k].astype(float)
	# features=np.array(features)
	features=np.nan_to_num(F)
	regis=np.array(regis)

	if grouped:

		print(features.shape, regis.shape)
		if aud:
			print('aud')
			inds=[]
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='id')['acronym']
				if 'AUD' in n:
					print(n)
					inds.append(i)
					print(inds)

			aud_reg=np.mean(features[:, inds, :], axis=1)
			aud_reg=np.expand_dims(aud_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, aud_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[ano.find('AUD', key='acronym')['id'], 7]]), axis=0)))#auditory areas
			aud=False

		print(features.shape, regis.shape)
		if vis:
			print('vis')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='id')['acronym']
				if 'VIS' in n:
					print(n)
					inds.append(i)
			vis_reg = np.mean(features[:, inds, :], axis=1)
			vis_reg = np.expand_dims(vis_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, vis_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[ano.find('VIS', key='acronym')['id'], 7]]), axis=0)))  # auditory areas
			vis = False

		print(features.shape, regis.shape)
		if rsp:
			print('rsp')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='id')['acronym']
				if 'RSP' in n:
					print(n)
					inds.append(i)
			rsp_reg = np.mean(features[:, inds, :], axis=1)
			rsp_reg = np.expand_dims(rsp_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, rsp_reg), axis=1)
			regis = np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[ano.find('RSP', key='acronym')['id'], 7]]), axis=0)))  # auditory areas
			rsp = False

		print(features.shape, regis.shape)
		if nose:
			print('nose')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='id')['acronym']
				if 'bfd' in n:
					inds.append(i)
				if 'SSp-n' in n:
					inds.append(i)
			nos_reg = np.mean(features[:, inds, :], axis=1)
			nos_reg = np.expand_dims(nos_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, nos_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[ano.find('SSp-n', key='acronym')['id'], 9]]), axis=0)) ) # auditory areas
			nose = False

		print(features.shape, regis.shape)
		if trunk:
			print('limbs')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='id')['acronym']
				if 'SSp-ll' in n:
					inds.append(i)
				if 'SSp-tr' in n:
					inds.append(i)
			nos_reg = np.mean(features[:, inds, :], axis=1)
			nos_reg = np.expand_dims(nos_reg, axis=1)
			features = np.delete(features, inds, axis=1)
			features = np.concatenate((features, nos_reg), axis=1)
			regis = np.delete(regis, inds, axis=0)
			regis = np.concatenate((regis, np.expand_dims(np.array([[ano.find('SSp-tr', key='acronym')['id'], 9]]), axis=0)))  # auditory areas
			trunk = False



	if brains==controls:
		features_avg=np.mean(features, axis=0)
		features_avg_c=features_avg
		features_c=features
	if brains==mutants:
		features_avg=np.mean(features, axis=0)
		features_avg_m=features_avg
		features_m=features






## plot Q and SBM nb clusters per regions
feat_list=[4,5]
for feat in feat_list:
	plt.figure()
	plt.xticks(np.arange(regis.shape[0]), [ano.find(regis[i][0][0], key='order')['acronym'] for i in range(regis.shape[0])])


	features_m[:,:, feat,:]
	for reg in range(features_m.shape[1]):
		array=features_m[:, reg, feat, :].flatten()
		plt.scatter(reg*np.ones(array.shape),array, c='indianred', alpha=0.5 )


	features_c[:,:, feat,:]
	for reg in range(features_c.shape[1]):
		array=features_c[:, reg, feat, :].flatten()
		plt.scatter(reg*np.ones(array.shape),array, c='cadetblue', alpha=0.5 )


def kolmogorov_distance(v1, v2):
  return np.linalg.norm(v1-v2)


feat_list=[1,2,3,4, 5, 6,7]
feat_list = [7]#[7,9]#3
# feat_list=[2, 3]
from sklearn.preprocessing import normalize
D=np.zeros((features_avg_m.shape[0],features_avg_m.shape[0]))
# w=[]
for r1 in range(features_avg_m.shape[0]):
  for r2 in range(features_avg_m.shape[0]):
    w=[]
    for f in range(features_avg_m.shape[1]):
      if f in feat_list:
        # mr1=np.sum(features_avg[r1, f])
        # mr2 = np.sum(features_avg[r2, f])
        n1=np.squeeze(normalize(features_avg_m[r1, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        n2=np.squeeze(normalize(features_avg_m[r2, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        w.append(kolmogorov_distance(n1,n2))#kolmogorov_distance#wasserstein_distance
        # print(f)
        # w.append(energy_distance(features_avg[r1, f],features_avg[r2, f]))

    D[r1,r2]=np.linalg.norm(np.array(w))


from scipy.stats import wasserstein_distance, energy_distance
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster, centroid
Z_m=centroid(D)#ward#centroid
N = len(D)
res_order = seriation(Z_m, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = D[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


fig = plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])
# plt.yticks(y_pos, layer_nb)

def lqbeling(id):
  order, level=regis[id][0]
  return ano.find(order, key='order')['acronym']

plt.subplot(1, 2, 2)
# fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z_m, orientation='right',leaf_label_func=lqbeling, leaf_font_size=12, color_threshold=1.3)
plt.title('mutants BP ORI')




from sklearn.preprocessing import normalize
D=np.zeros((features_avg_c.shape[0],features_avg_c.shape[0]))
# w=[]
for r1 in range(features_avg_c.shape[0]):
  for r2 in range(features_avg_c.shape[0]):
    w=[]
    for f in range(features_avg_c.shape[1]):
      if f in feat_list:
        # mr1=np.sum(features_avg[r1, f])
        # mr2 = np.sum(features_avg[r2, f])
        n1=np.squeeze(normalize(features_avg_c[r1, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        n2=np.squeeze(normalize(features_avg_c[r2, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        w.append(wasserstein_distance(n1,n2))#kolmogorov_distance#wasserstein_distance
        # print(f)
        # w.append(energy_distance(features_avg[r1, f],features_avg[r2, f]))

    D[r1,r2]=np.linalg.norm(np.array(w))


from scipy.stats import wasserstein_distance, energy_distance
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster, centroid
Z_c=centroid(D)#ward#centroid

N = len(D)
res_order = seriation(Z_c, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = D[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


# np.save('/mnt/data_SSD_2to/190408-44L/labelsMat.npy', labels)
# Z_c=Z


fig = plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(seriated_dist, cmap='cividis')
plt.xlim([0,N])
plt.ylim([0,N])
# plt.yticks(y_pos, layer_nb)

def lqbeling(id):
  order, level=regis[id][0]
  return ano.find(order, key='order')['acronym']

plt.subplot(1, 2, 2)
# fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z_c, orientation='right',leaf_label_func=lqbeling, leaf_font_size=12, color_threshold=0.045)
plt.title('controls BP ORI')







#### PCA
## fit control
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
clustering=True
avg=True

pca = PCA(n_components=2)#PCA(n_components=2)
if avg:
	X_avg=features_avg_c[:, :]
	X=features_c.reshape((features_c.shape[0]*features_c.shape[1], features_c.shape[2], features_c.shape[3]))
else:
	X_avg=features_c.reshape((features_c.shape[0]*features_c.shape[1], features_c.shape[2], features_c.shape[3]))
	X=X_avg
	
X_avg=X_avg.reshape((X_avg.shape[0], X_avg.shape[1]*X_avg.shape[2]))
X=X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
pca.fit(X_avg)
X_transform=pca.transform(X)

col_dict={1:'r', 2:'c', 3:'g', 4:'m', 5:'y'}

if clustering:
	clusters = fcluster(Z_c, 1.5, criterion='distance')

plt.figure()
sns.set_style(style='white')


for i, x in enumerate(X_transform):
	i = i % features_c.shape[1]
	if not clustering:
		col = ano.find(regis[i][0][0], key='order')['rgb']
	else:
		col = col_dict[clusters[i]]
	plt.scatter(x[1], x[0], c=col)
	label=ano.find(regis[i][0][0], key='order')['acronym']
	plt.text(x[1], x[0],label)  # horizontal alignment can be left, right or center
plt.title('cluster control')
sns.despine()

## transform deprived
if avg:
	Y_avg=features_avg_m[:, :]
	Y=features_m.reshape((features_m.shape[0]*features_m.shape[1], features_m.shape[2], features_m.shape[3]))
else:
	Y_avg=features_m.reshape((features_m.shape[0]*features_m.shape[1], features_m.shape[2], features_m.shape[3]))
	Y=Y_avg
# Y=features_avg_m[:, :]
Y_avg=Y_avg.reshape((Y_avg.shape[0], Y_avg.shape[1]*Y_avg.shape[2]))
Y=Y.reshape((Y.shape[0], Y.shape[1]*Y.shape[2]))
Y_transform=pca.transform(Y)

# col_dict={1:'r', 2:'g', 3:'c'}
if clustering:
	clusters = fcluster(Z_m, 1.5, criterion='distance')

plt.figure()
sns.set_style(style='white')
for i, x in enumerate(Y_transform):
	i = i % features_c.shape[1]
	if not clustering:
		col = ano.find(regis[i][0][0], key='order')['rgb']
	else:
		col = col_dict[clusters[i]]
	plt.scatter(x[1], x[0], c=col)

	label=ano.find(regis[i][0][0], key='order')['acronym']
	plt.text(x[1], x[0],label) # horizontal alignment can be left, right or center

# plt.scatter(Y_transform[:, 0], Y_transform[:, 1], c=clusters)
plt.title('cluster mutant')
sns.despine()

##########  plot comparison

ROI=['MOs', 'MOp', 'Alp', 'VIS', 'AUDp', 'AUDd', 'SSp-ul',  'SSp-ll', 'AUD', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]


X_transform_avg=pca.transform(features_avg_c.reshape((features_avg_c.shape[0], features_avg_c.shape[1]*features_avg_c.shape[2])))
f_cov=features_c.reshape((features_c.shape[0],features_c.shape[1], features_c.shape[2]*features_c.shape[3]))
X_transform_cov=np.array([np.cov(pca.transform(f_cov[:,i, :])[:,0],pca.transform(f_cov[:,i, :])[:,1]) for i in range(f_cov.shape[1])])

Y_transform_avg=pca.transform(features_avg_m.reshape((features_avg_m.shape[0], features_avg_m.shape[1]*features_avg_m.shape[2])))
f_cov=features_m.reshape((features_m.shape[0],features_m.shape[1], features_m.shape[2]*features_m.shape[3]))
Y_transform_cov=np.array([np.cov(pca.transform(f_cov[:,i, :])[:,0],pca.transform(f_cov[:,i, :])[:,1]) for i in range(f_cov.shape[1])])
# Y_transform_cov=pca.transform(np.cov(features_m, axis=0))


plt.figure()
ax = plt.gca()

fig, ax = plt.subplots()

# sns.set_style(style='white')
# col_dict={1:'skyblue', 2:'salmon', 3:'limegreen'}
col_dict={1:'r', 2:'c', 3:'g', 4:'m', 5:'y'}
clusters = fcluster(Z_c, 1.5, criterion='distance')
for i, x in enumerate(X_transform):
	i = i % features_c.shape[1]
	label = ano.find(regis[i][0][0], key='order')['acronym']
	col=col_dict[clusters[i]]
	col = 'cadetblue'
	if label in ROI:
		ax.scatter(x[1], x[0], c=col, alpha=1)
		ax.text(x[1], x[0],label,bbox=dict(facecolor=col, alpha=0.5))  # horizontal alignment can be left, right or center
	else:
		ax.scatter(x[1], x[0], c=col, alpha=0.3)
		# plt.text(x[0], x[1], label)  # horizontal alignment can be left, right or center

# col_dict={1:'firebrick', 2:'forestgreen', 3:'royalblue'}
# clusters = fcluster(Z, 3, criterion='distance')
for i, x in enumerate(Y_transform):
	i = i % features_c.shape[1]
	label = ano.find(regis[i][0][0], key='order')['acronym']
	col = col_dict[clusters[i]]
	col='indianred'
	if label in ROI:
		ax.scatter(x[1], x[0], c=col, alpha=1)
		ax.text(x[1], x[0], label, bbox=dict(facecolor=col, alpha=0.5))  # horizontal alignment can be left, right or center
	else:
		ax.scatter(x[1], x[0], c=col, alpha=0.3)
		# plt.text(x[0], x[1], label)  # horizontal alignment can be left, right or center




for i in range(26):
    # i = i % Y_transform_avg.shape[1]
    col='indianred'
    label = ano.find(regis[i][0][0], key='order')['acronym']
    print(i, label)
    if label in ROI:
        # ellipse=confidence_ellipse(pca.transform(f_cov[:,i, :])[:,0],pca.transform(f_cov[:,i, :])[:,1], ax)
        # ax.add_patch(ellipse)
        e = get_cov_ellipse(Y_transform_cov[i,:, :], (Y_transform_avg[i,1], Y_transform_avg[i,0]), 3,fc=col, alpha=0.4)
        ax.add_patch(e)



for i in range(26):
    # i = i % Y_transform_avg.shape[1]
    col='cadetblue'
    label = ano.find(regis[i][0][0], key='order')['acronym']
    print(i, label)
    if label in ROI:
        # ellipse=confidence_ellipse(pca.transform(f_cov[:,i, :])[:,0],pca.transform(f_cov[:,i, :])[:,1], ax)
        # ax.add_patch(ellipse)
        e = get_cov_ellipse(X_transform_cov[i,:, :], (X_transform_avg[i,1], X_transform_avg[i,0]), 3,fc=col, alpha=0.4)
        ax.add_patch(e)


plt.title('cluster comparison')
sns.despine()


####### brains comparison
X_transformed_brains=X_transform.reshape((4,52))
Y_transformed_brains=Y_transform.reshape((4,52))
# X_transformed_brains=features_c.reshape((4,1820))
# Y_transformed_brains=features_m.reshape((4,1820))
pca2 = PCA(n_components=2)  # PCA(n_components=2)



pca2.fit(X_transformed_brains)
X_transformed_brains_transform = pca2.transform(X_transformed_brains)
Y_transformed_brains_transform = pca2.transform(Y_transformed_brains)

col_dict = {1: 'r', 2: 'c', 3: 'g', 4: 'm', 5: 'y'}


plt.figure()
sns.set_style(style='white')

for i, x in enumerate(X_transformed_brains_transform):
	col = 'cadetblue'
	plt.scatter(x[1], x[0], c=col)
	label = ano.find(regis[i][0][0], key='order')['acronym']
	plt.text(x[1], x[0], controls[i])  # horizontal alignment can be left, right or center

for i, x in enumerate(Y_transformed_brains_transform):
	col = 'indianred'
	plt.scatter(x[1], x[0], c=col)
	label = ano.find(regis[i][0][0], key='order')['acronym']
	plt.text(x[1], x[0], mutants[i])  # horizontal alignment can be left, right or center


plt.title('cluster control')
sns.despine()


orientation_method='flowInterpolation'#'localNormal'

# regions=[[(6,6)]]
# regions=[[(54, 9), (47, 9)]]
regions=[[(19, 8)]]
# regions=[[(142, 8), (149, 8), (128, 8), (156, 8)]]

BP=[]
Orientations=[]
dist_2surface=[]
for control in controls:
	graph = ggt.load(
        work_dir + '/' + control + '/' + '/data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
	degrees = graph.vertex_degrees()
	vf = np.logical_and(degrees > 1, degrees <= 4)
	graph = graph.sub_graph(vertex_filter=vf)
	label = graph.vertex_annotation();
	# vertex_filter = from_e_prop2_vprop(graph, 'artery')
	# art_tree = graph.sub_graph(vertex_filter=vertex_filter)
	# art_tree = graph.sub_graph(vertex_filter=vertex_filter)
	with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
		sampledict = pickle.load(fp)

	pressure = np.asarray(sampledict['pressure'][0])
	graph.add_vertex_property('pressure', pressure)

	for region_list in regions:

		vertex_filter = np.zeros(graph.n_vertices)
		for i, rl in enumerate(region_list):
			order, level = region_list[i]
			print(level, order, ano.find(order, key='order')['name'])
			label = graph.vertex_annotation();
			label_leveled = ano.convert_label(label, key='order', value='order', level=level)
			vertex_filter[label_leveled == order] = 1;
		gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

		gss4 = gss4_t.copy()
		nv = gss4.n_vertices
		ne = gss4.n_edges

		if orientation_method=='localNormal':
			r, p, l = getRadPlanOrienttaion(gss4, gss4_t,local_normal=True, verbose=True)  # , local_normal=True, calc_art=True)
			r = r[~np.isnan(r)]
			p = p[~np.isnan(r)]
			dist=gss4.edge_property('distance_to_surface')[~np.isnan(r)]
			dist_2surface.append(dist)
			radiality = (r / (r + p)) > 0.5
			planarity = (p / (r + p)) > 0.6
		elif orientation_method=='flowInterpolation':
			angle=GeneralizedRadPlanorientation(gss4, control)
			dist = gss4.edge_property('distance_to_surface')
			dist_2surface.append(dist)
			radiality = angle < 40
			planarity = angle > 50

		neutral = np.logical_not(np.logical_or(radiality, planarity))
		bp=gss4.vertex_property('distance_to_surface')
		ori=np.concatenate((np.expand_dims(radiality, axis=1),np.concatenate((np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)
		print(ori.shape)
		Orientations.append(ori)
		BP.append(bp)

import pandas as pd
colors = ['indianred', 'tan', 'forestgreen']
plt.figure()
sns.set_style(style='white')
# sns.despine()
P=np.zeros((len(controls), 10))
N=np.zeros((len(controls), 10))
R=np.zeros((len(controls), 10))
BP=np.zeros((len(controls), 10))

for i in range(len(controls)):
	print(i)
	print(controls[i])
	dist=dist_2surface[i]
	ori_rad=dist[Orientations[i][:,0]]
	ori_neutral=dist[Orientations[i][:,1]]
	ori_plan=dist[Orientations[i][:,2]]
	plt.hist([ori_rad, ori_neutral, ori_plan], bins=10, histtype='bar', color=colors, stacked=True, alpha=0.1)
	histrad, bins=np.histogram(ori_rad,bins=10)
	histneut, bins=np.histogram(ori_neutral,bins=bins)
	histplan, bins=np.histogram(ori_plan,bins=bins)
	histbp, bins=np.histogram(dist,bins=10)
	R[i]=histrad/(histrad+histneut+histplan)
	N[i]=histneut/(histrad+histneut+histplan)
	P[i]=histplan/(histrad+histneut+histplan)
	BP[i]=histbp
	print(BP)

plt.yticks(size='x-large')
plt.yticks(size='x-large')
plt.xlabel('cortical depth')
ax1t = plt.twinx()

x=(bins[:-1]+bins[1:])/2
X=np.array([x,x,x,x]).flatten()

histrad = pd.DataFrame(np.array(R).transpose()).melt()
histrad['variable']=X
# histrad=np.mean(R, axis=0)
histneut = pd.DataFrame(np.array(N).transpose()).melt()
histneut['variable']=X
# histneut=np.mean(N, axis=0)
histplan = pd.DataFrame(np.array(P).transpose()).melt()
histplan['variable']=X
# histplan=np.mean(P, axis=0)
histbp = pd.DataFrame(np.array(BP).transpose()).melt()
histbp['variable']=X


# sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=histrad, color='indianred')
# sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=histplan, color='forestgreen')
# sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=histneut , color='black')
plt.xticks(size='x-large')
sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=histbp , color='black')
# plt.legend(['radial', 'planar','neutral', 'branching point'])
plt.legend(['branching point'])
ax1t.set_ylim([0,20000])
plt.title('Orientation and branching point distribution across cortex', size='x-large')





################ plot individual values for ROI
ROI=['MOs', 'MOp', 'Alp', 'VIS', 'AUD', 'AUDd', 'SSp-ul',  'SSp-ll', 'AUD', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]

ROI=['MOs', 'MOp', 'SSp-m', 'VIS', 'AUD', 'SSp-ul',  'SSp-ll', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]
ROI=['AUD']
ROI=['VIS','AUD','MOs', 'MOp', 'SSp-n' ]
ROI=['AUDp', 'VISp', 'MOs', 'MOp', 'SSp-n','SSp-tr','SSs']
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

ROI=['AUDp', 'AUDd', 'AUDv', 'AUDpo']

colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange','forestgreen', 'lightseagreen']
# ROI=['SSp-n']#['SSp-tr', 'MOs', 'MOp', 'SSp-n', 'SSp-bfd','SSs']
norm=False
f=2
# bins=bins_bp_dist
feat=['BP','PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']#['ART EP', 'VE EP', 'BP', 'ORI']#'SP len', 'SP step',
import pandas as pd
for r in range(features.shape[1]):
    l = ano.find(regis[r][0][0], key='id')['acronym']
    print(l)
    # print(l)
    if l in ROI:
        print(l, r)
        plt.figure()
        sns.set_style(style='white')

        # for b in range(features.shape[0]):

		### normal features
		# plt.plot(np.squeeze(normalize(features_c[b, r, f].reshape(-1, 1), norm='l2', axis=0), axis=1), color=colors_c[b])
		# plt.plot(np.squeeze(features[b, r, f].reshape(-1, 1), axis=1), color=colors_m[b])
		# plt.title(l + ' ' + feat[0], size='x-large')
		# for i in range(5):
		norm = False
		f = 8#3
		if norm:
			Cpd_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
			# Cpd_m = pd.DataFrame(normalize(features[:, r, f-2], norm='l2', axis=1)).melt()
			Cpd_m = pd.DataFrame(normalize(features_m[:, r, f ], norm='l2', axis=1)).melt()
		else:
			Cpd_c = pd.DataFrame(features_c[:, r, f]).melt()
			# Cpd_m = pd.DataFrame(features[:, r, f-2]).melt()
			Cpd_m = pd.DataFrame(features_m[:, r, f]).melt()

		f =  2
		norm = False
		if norm:
			bp_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
			# bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
			bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
		else:
			bp_c = pd.DataFrame(features_c[:, r, f]).melt()
			bp_m = pd.DataFrame(features_m[:, r, f]).melt()


			## art/vein ep features
			# if norm:
			#   Cpd_art_c = pd.DataFrame(normalize(np.array(features_c[:, r, 0]), norm='l2', axis=1)).melt()
			#   Cpd_art_m = pd.DataFrame(normalize(np.array(features[:, r, 0]), norm='l2', axis=1)).melt()
			#   Cpd_ve_c = pd.DataFrame(normalize(np.array(features_c[:, r, 1]), norm='l2', axis=1)).melt()
			#   Cpd_ve_m = pd.DataFrame(normalize(np.array(features[:, r, 1]), norm='l2', axis=1)).melt()
			# else:
			#   Cpd_art_c = pd.DataFrame(np.array(features_c[:, r, 0])).melt()
			#   Cpd_art_m = pd.DataFrame(np.array(features[:, r, 0])).melt()
			#   Cpd_ve_c = pd.DataFrame(np.array(features_c[:, r, 1])).melt()
			#   Cpd_ve_m = pd.DataFrame(np.array(features[:, r, 1])).melt()
			# sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_art_c, color=colors_m[0])
			# sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_art_m, color=colors_m[3])
			# sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_ve_c, color=colors_c[0])
			# sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_ve_m, color=colors_c[3])
			# plt.title(l+' '+'ART/VEIN EP')
			# plt.legend(['art_c', 'art_v', 've_c', 've_m'])
			# sns.despine()



			sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=colors_c[0], linewidth=2.5)
			sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color=colors_m[1], linewidth=2.5)
			# plt.legend(['prop rad control', 'prop rad mutant'])#(['prop rad', 'prop plan'])
			plt.yticks(size='x-large')
			plt.ylabel(feat[2], size='x-large')
			plt.xlabel('cortical depth (um)', size='x-large')
			plt.xticks(size='x-large')
			plt.xticks(np.arange(0, 10), 25*np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large', rotation=20)
			plt.twinx()
			sns.lineplot(x="variable", y="value", err_style='bars', data=bp_c, color=colors_c[2], linewidth=2.5)
			sns.lineplot(x="variable", y="value", err_style='bars', data=bp_m, color=colors_m[3], linewidth=2.5)


		plt.title(l + ' ' + feat[0], size='x-large')
		plt.legend(['bp control', 'bp mutant'])#(['bp'])#['control', 'deprived'])
		plt.yticks(size='x-large')
		plt.ylabel(feat[0], size='x-large')


		plt.vlines(np.array([150, 550, 800, 1300])/250,0, np.max(bp_m)['value'], colors='k', linestyles='dashed')

		# sns.despine()
		plt.tight_layout()
		# plt.legend(['bp control', 'bp mutants'])
