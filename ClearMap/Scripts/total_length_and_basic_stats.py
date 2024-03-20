from general_fonction import *

work_dir='/data_SSD_2to/211019_otof_10m'
mutants=['1k', '2k','3k', '6k']#456 not annotated ?
controls=['7w', '9w', '10w', '12w', '13w']
states=[controls, mutants]



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
    region_list = [(6, 6)]  # isocortex
    regions = []
    R = ano.find(region_list[0][0], key='order')['name']
    main_reg = region_list
    sub_region = True
    for r in reg_list.keys():
        l = ano.find(r, key='order')['level']
        regions.append([(r, l)])



# regions = [[(54, 9)], [(25, 8)]]



compute_SBM=False
compute_flow=False
compute_prop_ori=True
orientation_method = 'flowInterpolation'#'localNormal'#flowInterpolation
limit_angle=40
mode='bigvessels'
average=True


BP_controls=[]
BP_mutants=[]

EP_controls=[]
EP_mutants=[]

ORI_controls=[]
ORI_mutants=[]

ARTBP_controls=[]
ARTBP_mutants=[]

VDIST_controls=[]
VDIST_mutants=[]

EDIST_controls=[]
EDIST_mutants=[]

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
		graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
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
				order, level = region_list[i]
				print(level, order, ano.find(order, key='order')['name'])
				label = graph.vertex_annotation();
				label_leveled = ano.convert_label(label, key='order', value='order', level=level)
				vertex_filter[label_leveled == order] = 1;
			gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
		
			gss4=gss4_t.copy()
			nv=gss4.n_vertices
			ne=gss4.n_edges
			bp_dist_2_surface.append(gss4.vertex_property('distance_to_surface'))
			bp.append(gss4.vertex_property('distance_to_surface'))
			ep.append(gss4.edge_property('distance_to_surface'))
			print('compute orientation ...')
			if orientation_method == 'localNormal':
				try:
					r, p, n, l = getRadPlanOrienttaion(gss4, gss4_t,local_normal=True, calc_art=False) # ,, , calc_art=True)

					if not compute_prop_ori:
						r = r[~np.isnan(r)]
						p = p[~np.isnan(r)]
					elif compute_prop_ori:
						r = r[~np.isnan(r)]
						p = p[~np.isnan(r)]
						dist = gss4.edge_property('distance_to_surface')[~np.isnan(r)]

						# radiality = (r / (r + p)) > 0.5
						# planarity = (p / (r + p)) > 0.6
						# neutral = np.logical_not(np.logical_or(radiality, planarity))
						angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

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
		vess_rad_control=np.load(work_dir + '/' + 'ORI_fi_bv' + condition +  '_'+control+'.npy',allow_pickle=True)
		prop_ori_control=np.load(work_dir + '/' + 'PROP_ORI_fi_bv' + condition + '_' + control + '.npy',allow_pickle=True)
		bp_dist_2_surface_control=np.load(work_dir + '/' + 'BP_' + condition +  '_'+control+'.npy',allow_pickle=True)
		# art_ep_dist_2_surface_control = np.load(work_dir + '/' + 'ARTBP_' + condition + '_' + control + '.npy',allow_pickle=True)
		ve_ep_dist_2_surface_control = np.load(work_dir + '/' + 'EP_' + condition + '_' + control + '.npy', allow_pickle=True)


		features_brains = []
		for reg in range(nb_reg):
			# depth = []
			order, level = regions[reg][0]
			n = ano.find(order, key='order')['acronym']
			id=ano.find(order, key='order')['id']
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
						features_brains.append([np.zeros(10),np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)])
		
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
				n = ano.find(order, key='order')['acronym']
				if 'AUD' in n:
					print(n)
					inds.append(i)
					print(inds)

			aud_reg=np.mean(features[:, inds, :], axis=1)
			aud_reg=np.expand_dims(aud_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, aud_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[127, 7]]), axis=0)))#auditory areas
			aud=False

		print(features.shape, regis.shape)
		if vis:
			print('vis')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='order')['acronym']
				if 'VIS' in n:
					print(n)
					inds.append(i)
			vis_reg = np.mean(features[:, inds, :], axis=1)
			vis_reg = np.expand_dims(vis_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, vis_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[163, 7]]), axis=0)))  # auditory areas
			vis = False

		print(features.shape, regis.shape)
		if rsp:
			print('rsp')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='order')['acronym']
				if 'RSP' in n:
					print(n)
					inds.append(i)
			rsp_reg = np.mean(features[:, inds, :], axis=1)
			rsp_reg = np.expand_dims(rsp_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, rsp_reg), axis=1)
			regis = np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[303, 7]]), axis=0)))  # auditory areas
			rsp = False

		print(features.shape, regis.shape)
		if nose:
			print('nose')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='order')['acronym']
				if 'bfd' in n:
					inds.append(i)
				if 'SSp-n' in n:
					inds.append(i)
			nos_reg = np.mean(features[:, inds, :], axis=1)
			nos_reg = np.expand_dims(nos_reg, axis=1)
			features=np.delete(features, inds, axis=1)
			features=np.concatenate((features, nos_reg), axis=1)
			regis=np.delete(regis, inds, axis=0)
			regis=np.concatenate((regis, np.expand_dims(np.array([[47, 9]]), axis=0)) ) # auditory areas
			nose = False

		print(features.shape, regis.shape)
		if trunk:
			print('nose')
			inds = []
			for i, r in enumerate(regis):
				order, level = r[0]
				n = ano.find(order, key='order')['acronym']
				if 'SSp-ll' in n:
					inds.append(i)
				if 'SSp-tr' in n:
					inds.append(i)
			nos_reg = np.mean(features[:, inds, :], axis=1)
			nos_reg = np.expand_dims(nos_reg, axis=1)
			features = np.delete(features, inds, axis=1)
			features = np.concatenate((features, nos_reg), axis=1)
			regis = np.delete(regis, inds, axis=0)
			regis = np.concatenate((regis, np.expand_dims(np.array([[89, 9]]), axis=0)))  # auditory areas
			trunk = False



	if brains==controls:
		features_avg=np.mean(features, axis=0)
		features_avg_c=features_avg
		features_c=features
	if brains==mutants:
		features_avg=np.mean(features, axis=0)
		features_avg_m=features_avg
		features_m=features






################ plot individual values for ROI
ROI=['MOs', 'MOp', 'Alp', 'VIS', 'AUD', 'AUDd', 'SSp-ul',  'SSp-ll', 'AUD', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]

ROI=['MOs', 'MOp', 'SSp-m', 'VIS', 'AUD', 'SSp-ul',  'SSp-ll', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]
ROI=['AUD']
ROI=['VISp','AUDp','MOs', 'MOp', 'SSp-n','AUD' ]
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange','forestgreen', 'lightseagreen']
# ROI=['SSp-n']#['SSp-tr', 'MOs', 'MOp', 'SSp-n', 'SSp-bfd','SSs']
norm=False
f=2
# bins=bins_bp_dist
feat=['BP','PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']#['ART EP', 'VE EP', 'BP', 'ORI']#'SP len', 'SP step',
import pandas as pd
for r in range(features.shape[1]):
    l = ano.find(regis[r][0][0], key='order')['acronym']
    print(l)
    # print(l)
    if l in ROI:
        print(l)
        plt.figure()
        sns.set_style(style='white')

        # for b in range(features.shape[0]):

		
	norm = False
	f = 8#3 ORI
	if norm:
		Cpd_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
		# Cpd_m = pd.DataFrame(normalize(features[:, r, f-2], norm='l2', axis=1)).melt()
		Cpd_m = pd.DataFrame(normalize(features_m[:, r, f ], norm='l2', axis=1)).melt()
	else:
		Cpd_c = pd.DataFrame(features_c[:, r, f]).melt()
		# Cpd_m = pd.DataFrame(features[:, r, f-2]).melt()
		Cpd_m = pd.DataFrame(features_m[:, r, f]).melt()

	f =  2# BP
	norm = False
	if norm:
		bp_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
		bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
		# bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
	else:
		bp_c = pd.DataFrame(features_c[:, r, f]).melt()
		bp_m = pd.DataFrame(features_m[:, r, f]).melt()



	sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=colors_c[0], linewidth=2.5)
	sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color=colors_m[1], linewidth=2.5)
	plt.legend(['prop rad control', 'prop rad mutant'])#(['prop rad', 'prop plan'])
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



