
from sklearn.preprocessing import normalize
import pandas as pd
import seaborn as sns
# brains=['11L', '12L', '13L', '14L', '15L', '16L', '17L', '18L', '19L', '20L', '21L']


work_dir='/data_SSD_2to/cFosDeprived/3weeks-secondbatch/'
region = [(54, 9), (47, 9)]  # NOSE
R=[]
atlas_R=io.read(io.join(work_dir, 'Annotation-180305-FOS-R.tif'))
atlas_L=io.read(io.join(work_dir, 'Annotation-180305-FOS-L.tif'))
mask_R=np.zeros(atlas_R.shape)
mask_L=np.zeros(atlas_L.shape)
distance_R=io.read(io.join(work_dir,'Annotation_180305_FOS_distance_R.tif'))
distance_L=io.read(io.join(work_dir,'Annotation_180305_FOS_distance_L.tif'))
bin=10
shape=mask_R.shape


normed=False
colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']


for reg in region:
    order, level = reg
    layers = ['1', '2/3', '4', '5', '6a', '6b']
    # layers = ['4']
    for b, layer in enumerate(layers):
        R = ano.find(order, key='order')['name']
        for r in reg_list.keys():
            n = ano.find_name(r, key='order')
            if R in n:
                for se in reg_list[r]:
                    if layer in ano.find(se, key='order')['name']:
                        print(ano.find(se, key='order')['name'])
                        mask_R[atlas_R==ano.find(se, key='order')['id']]=1
                        mask_L[atlas_L == ano.find(se, key='order')['id']] = 1




work_dir='/data_SSD_2to/cFosDeprived/PositiveControl/'#'/data_SSD_2to/cFosDeprived/3weeks-secondbatch/'#'/data_SSD_2to/cFosDeprived/PositiveControl/'
# brains=['1-L', '7-R', '10-R', '4-R', '5-L', '6-L']
brains=['1L', '3L', '4L', '5L', '5L', '6L', '7L', '8L', '9L', '11L', '12L']

activity_profile=[]
for b in brains:
    print(b)
    if 'L' in b:
        mask=mask_L
        atlas=atlas_L
        distance=distance_L
        print('L in ')
    elif 'R' in b:
        mask=mask_R
        atlas=atlas_R
        distance = distance_R
        print('R in ')
    else:
        print('name sample pb !')
    brain_activity=[]
    cells= np.load(work_dir+b+'-cells_transformed_to_Atlas.npy')
    try:
        for c in cells:
            if mask[ min(int(c[0]), shape[0]-1), min(int(c[1]), shape[1]-1), min(int(c[2]), shape[2]-1)]:
                brain_activity.append(distance[min(int(c[0]), shape[0]-1), min(int(c[1]), shape[1]-1), min(int(c[2]), shape[2]-1)])
    except IndexError:
        print(b,c, 'out of range')
    print(brain_activity)
    activity_profile.append(brain_activity)


cfos=[]
for a in activity_profile:
    hist_r, bins_r = np.histogram(a, bins=bin, normed=normed)
    cfos.append(hist_r)

cfos=np.array(cfos)
cfos=np.nan_to_num(cfos)

# control=[0,2,3,4,5]
# deprived=[6,7,8,9,10]
deprived=[0,1,2,3,4,5]
control=[6,7,8,9,10,]

plt.figure()

if normed:
    Cpd_all = pd.DataFrame(normalize(cfos[control, :], norm='l2', axis=1)).melt()
else:
    Cpd_all = pd.DataFrame(cfos[control, :]).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_all, color=colors_c[0])

if normed:
    Cpd_all = pd.DataFrame(normalize(cfos[deprived, :], norm='l2', axis=1)).melt()
else:
    Cpd_all = pd.DataFrame(cfos[deprived, :]).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_all, color=colors_m[1])

plt.title('cfos VS depth positiveControl', size='xx-large')
plt.xticks(size='xx-large')
plt.yticks(size='xx-large')
plt.xlabel('depth', size='xx-large')
plt.ylabel('normalized cfos', size='xx-large')
plt.legend(['c', 'd'])
sns.despine()

plt.figure()
for c in cfos:
    # Cpd_all = pd.DataFrame(normalize(c.reshape(-1, 1), norm='l2', axis=1)).melt()
    plt.plot(c)
    # sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_all, color=colors_c[0])

plt.title('cfos VS depth')
plt.legend(brains)

