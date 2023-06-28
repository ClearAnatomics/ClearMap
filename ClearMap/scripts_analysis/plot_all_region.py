














D1=pd.read_csv('/data_2to/dev/BP_smoothed_all_brain_metaregion.csv')



df=D1.copy()


for col in df.columns[[1, 2, 3, 4, 5,  7]]:
    # for col in df.columns[:-1]:
    new_col=df[col].values.astype(int)
    df[col]=new_col


var1='nbdeg1' #'lengthtot'
var2='lengthtot'

vars=[var1, var2]

r_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Brain stem', 'Striatum',
        'Midbrain', 'Olfactory areas', 'Hippocampal region', 'Somatomotor areas',
        'Primary somatosensory area, barrel field','Auditory areas',
        'Ventral posteromedial nucleus of the thalamus',
        'Medial geniculate complex',"Superior olivary complex","Inferior colliculus"]

r_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Brain stem', 'Striatum','Midbrain','Hippocampal region']


r_name=['Somatomotor areas',
          'Primary somatosensory area, barrel field','Auditory areas',
          'Visual areas']


r_name=['Medial geniculate complex',"Superior olivary complex","Inferior colliculus",'Ventral posteromedial nucleus of the thalamus']


palette=[ano.find(rn, key='name')['rgb'] for rn in r_name]

dfmax_reg=df.groupby('region').max()
dfmin_reg=df.groupby('region').min()
dfmean_reg=df.groupby(['region', 'timepoint']).mean()

dfmean_reg=df.groupby(['region', 'timepoint'], as_index=False).mean().groupby(['region']).max()
timepoint_temp=df['timepoint']

df_temp=df.copy()
for var in vars:
    L=df[var].array
    R=df['region'].array

    NL=[L[i]/dfmean_reg[var][R[i]] for i in range(L.shape[0]) ]


    df_temp['normalized_'+var]=NL
    # df_temp=(df[['region', 'timepoint','lengthtot']].set_index('region')/dfmean_reg['lengthtot']).reset_index()
    df_temp['timepoint']=timepoint_temp

#
#
# sns.lineplot(x="timepoint", y='lengthtot',err_style='bars',dashes=True, estimator = lambda x: (np.mean(x)-dfmin_reg.loc[rs, 'lengthtot'])/(dfmax_reg.loc[rs, 'lengthtot']-dfmin_reg.loc[rs, 'lengthtot']), color='darkorchid',
#              data=D1[D1['region']==rs])



var1='nbdeg1' #'lengthtot'
var2='lengthtot'

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df_temp, row="region", hue="region", aspect=15, height=.5, palette=palette)
# Draw the densities in a few steps
# g.map(sns.lineplot, "timepoint",'normalized_'+var1, alpha=1, linewidth=1.5, color='grey')
g.map(sns.lineplot, "timepoint",'normalized_'+var2, alpha=1, linewidth=1.5)
# g.map(sns.lineplot, "timepoint",var, alpha=1, linewidth=1.5)
# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
g.map(label, "timepoint")
# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.15)
# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)