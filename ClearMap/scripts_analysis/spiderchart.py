# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import seaborn as sns

# Set data
df = pd.DataFrame({
    'group': ['A','B','C','D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
})

df=D1

# ------- PART 1: Create background
TPs=[1,3,5,7, 12, 14,21, 30]
# val_min=10
# val_max=20
# step=3
val_min=0
val_max=50000
step=5000
pal = sns.cubehelix_palette(len(TPs), rot=-.25, light=7.)

TPs=[21, 30, 60, 90, 210]
# val_min=10
# val_max=20
# step=3
val_min=0
val_max=60000
step=10000
pal = sns.color_palette("mako", n_colors=len(TPs))

x='lengthtot'#'n_vertices/length'#
y='region'

plt.figure()

# number of variable
categories=list(df)[1:]
N = len(categories)

categories=np.unique(list(df['region']))
idx = [0, 3, 1, 2, 4, 5]
categories=categories[idx]#put isocortex and brainstem close together

N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
# plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
# plt.ylim(0,40)

plt.yticks(list(np.arange(val_min, val_max+step, step)), np.arange(val_min, val_max+step, step).astype('str'), color="grey", size=7)
plt.ylim(val_min,val_max)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable


# Ind1
# values=df.loc[0].drop('group').values.flatten().tolist()
# values += values[:1]


for i, tp in enumerate(TPs):
    values=df[df['timepoint']==tp]
    # x = df.get(x, x)
    # y = df.get(y, y)

    values=values.groupby(y)[x].mean().values.tolist()
    values=np.array(values)[idx].tolist()
    values.append(values[0])
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=str(tp), color=pal[i])
    # ax.fill(angles, values, 'b', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the graph
plt.show()

