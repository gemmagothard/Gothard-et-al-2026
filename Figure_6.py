import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code_utils import analysis_plotting as pl
import numpy as np
import figure1_utils as f1_ut
from matplotlib.patches import Circle
import f6_utils as f6_ut

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

colors = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }

condition_mapping = {
        'Ta1 Cre-OFF':'OP-der',
        'Ta1 Cre-ON': 'IP-der',
        }

PV_csv = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/PV_input_results.csv')
PV_csv = PV_csv[PV_csv.id=='pv_inputs']
starter_csv = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/starter_cell_results.csv')
simulated_data = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/simulated_results.csv') 


PV_csv = PV_csv[PV_csv.condition.isin(colors.keys())]
starter_csv  = starter_csv[starter_csv.condition.isin(colors.keys())]

PV_csv_describe = PV_csv.groupby(['condition'])['distance_from_mean_starter'].describe()
PV_csv_describe["sem"] = (
    PV_csv.groupby("condition")["distance_from_mean_starter"]
    .sem()
)
print(PV_csv_describe[['count','mean','sem']])

starter_spread_by_brain = (
    starter_csv
    .groupby(['condition', 'brain_ID'])['starter_dist_to_center']
    .std()
    .reset_index(name='starter_spread'))
IP_mean_locality_boundary = starter_spread_by_brain[starter_spread_by_brain.condition=='IP-der'].starter_spread.mean() * 2 
OP_mean_locality_boundary = starter_spread_by_brain[starter_spread_by_brain.condition=='OP-der'].starter_spread.mean() * 2


# Add brain-specific starter spread to each PV input
PV_csv['layer'] = PV_csv['layer'].str.replace('Layer ', 'L', regex=False)
pv_inputs = PV_csv.merge(
    starter_spread_by_brain,
    on=['condition', 'brain_ID'],
    how='left'
)
# Classify PV inputs
pv_inputs['locality'] = np.where(
    pv_inputs['distance_from_mean_starter'] > pv_inputs['starter_spread']*2,
    'non-local',
    'local'
)
locality_fraction = (
    pv_inputs
    .groupby(['condition', 'brain_ID', 'locality'])
    .size()
    .reset_index(name='count')
)
# Fraction within each brain
locality_fraction['fraction'] = (
    locality_fraction['count']
    / locality_fraction.groupby(['condition', 'brain_ID'])['count'].transform('sum') *100
)
local_fraction = locality_fraction[locality_fraction.locality=='local']
g = local_fraction.groupby("condition")["fraction"]
fraction_describe = g.describe()
fraction_describe["sem"] = g.sem()
print(fraction_describe[['mean','count','sem']])


counts = PV_csv.groupby(['condition','brain_ID','layer'])['pv_id'].count()
layers = ['L2/3', 'L4', 'L5a', 'L5b', 'L6']
# Add missing layers as 0
counts = (
    counts
    .unstack('layer')
    .reindex(columns=layers, fill_value=0)
    .fillna(0)
    .stack()
)
# Convert counts to percentage within each brain
percent = (
    counts
    / counts.groupby(['condition', 'brain_ID']).transform('sum')
    * 100
).rename('percent').reset_index()


# MAKE FIGURE - A4 portrait
fig = plt.figure(figsize=(8.9, 11.69))
# Main grid: 3 rows
n_rows = 4
n_cols = 1
gs = fig.add_gridspec(
    n_rows, n_cols,
    top=0.97,
    left=0.1,
    right=0.9,
    bottom=0.1,
    hspace=0.4,
    height_ratios=[0.9,1,1,1.2])
x_offset = -0.07
# top row
gs_1 = gs[0].subgridspec(1, 3, wspace=0.2, width_ratios=[1,1,0.5])
empty1 = fig.add_subplot(gs_1[0, 0])
empty2 = fig.add_subplot(gs_1[0, 1])
ax_layers = fig.add_subplot(gs_1[0, 2])
empty1.axis('off')
empty2.axis('off')
pl.add_panel_letter(fig, gs_1[0, 0], "a", x_offset=x_offset)
pl.add_panel_letter(fig, gs_1[0, 1], "b",x_offset=x_offset)
pl.add_panel_letter(fig, gs_1[0, 2], "c",x_offset=-0.05)

# second row
gs_2 = gs[1].subgridspec(1, 3, wspace=0.2, width_ratios=[1,1,0.02])
OP_der_ax = fig.add_subplot(gs_2[0, 0])
IP_der_ax = fig.add_subplot(gs_2[0, 1],
                            sharey=OP_der_ax,
                            sharex=OP_der_ax)
empty = fig.add_subplot(gs_2[0, 2])
empty.axis('off')
pl.add_panel_letter(fig, gs_2[0, 0], "d",x_offset=x_offset)

# third row
gs_3 = gs[2].subgridspec(1, 3, wspace=0.4)
ecdf_pv_dist = fig.add_subplot(gs_3[0, 0])
bar_pv_dist = fig.add_subplot(gs_3[0, 1])
bar_locality = fig.add_subplot(gs_3[0, 2])
pl.add_panel_letter(fig, gs_3[0, 0], "e",x_offset=x_offset)
pl.add_panel_letter(fig, gs_3[0, 1], "f",x_offset=x_offset)
pl.add_panel_letter(fig, gs_3[0, 2], "g",x_offset=x_offset)

# fourth row
gs_4 = gs[3].subgridspec(1, 4, width_ratios=[1,1,1,1])
# Split gs_lowest[0, 0] into 2 rows
examples_ar = gs_4[0, 0].subgridspec(2, 1)
example_ar_top = fig.add_subplot(examples_ar[0, 0])
example_ar_bottom = fig.add_subplot(examples_ar[1, 0])
# Split gs_lowest[0, 2] into 2 rows
examples_mode = gs_4[0, 2].subgridspec(2, 1)
example_mode_top = fig.add_subplot(examples_mode[0, 0])
example_mode_bottom = fig.add_subplot(examples_mode[1, 0])
pl.add_panel_letter(fig, gs_4[0, 0], "h",x_offset=x_offset)
hist_ar =  fig.add_subplot(gs_4[0, 1])
hist_mode = fig.add_subplot(gs_4[0, 3])
for ax in [example_ar_top, example_ar_bottom,
           example_mode_top, example_mode_bottom]:
    ax.axis('off')
    ax.margins(0)
    ax.set_anchor('W')
for ax in [example_ar_top, example_ar_bottom,
           example_mode_top, example_mode_bottom, hist_ar, hist_mode]:
    pos = ax.get_position()
    ax.set_position([
        pos.x0 - 0.04,  # Move left
        pos.y0,
        pos.width,
        pos.height
    ])

sns.barplot(data=percent,
            x='percent',
            y='layer',
            hue='condition',
            palette=colors,
            ax=ax_layers,
            errorbar='se',
            legend=False)
ax_layers.set_ylabel('')
ax_layers.set_xlabel('PV input cells (%)')
import pingouin as pg
anova = pg.mixed_anova(
    data=percent,
    dv='percent',
    within='layer',
    between='condition',
    subject='brain_ID',
    correction='auto'
)
print(anova)


sns.kdeplot(PV_csv[PV_csv.condition=='OP-der'],
            x='X_rel',
            y='Y',
            hue='condition',
            palette=colors,
            fill=True,
            ax=OP_der_ax,
            legend=False,
            rasterized=True,
            levels=100
            )
OP_der_ax.set_aspect("equal", adjustable="box")
starter_OP = starter_csv[starter_csv.condition == "OP-der"]
OP_der_ax.errorbar(
    0, starter_OP.Y.mean(),
    xerr=starter_OP.X_rel.std(),
    yerr=starter_OP.Y.std(),
    fmt="o",
    color="k",
    capsize=3,
    zorder=10,
)
from matplotlib.lines import Line2D
# Custom legend handle: dot only
mean_handle = Line2D(
    [0], [0],
    marker="o",
    linestyle="None",
    color="k",
    label="Mean starter \nneuron position"
)
IP_der_ax.legend(handles=[mean_handle],
                frameon=False,
                loc="upper right",
                bbox_to_anchor=(1.2, 1)
                 )


sns.kdeplot(PV_csv[PV_csv.condition=='IP-der'],
            x='X_rel',
            y='Y',
            hue='condition',
            palette=colors,
            fill=True,
            ax=IP_der_ax,
            legend=False,
            rasterized=True,
            levels=100
            )
IP_der_ax.set_aspect("equal", adjustable="box")
starter_IP = starter_csv[starter_csv.condition == "IP-der"]
IP_der_ax.errorbar(
    0, starter_IP.Y.mean(),
    xerr=starter_IP.X_rel.std(),
    yerr=starter_IP.Y.std(),
    fmt="o",
    color="k",
    capsize=3,
    zorder=10
)
circle = Circle(
    (0, starter_IP.Y.mean()),
    radius=IP_mean_locality_boundary,
    fill=False,
    edgecolor='k',
    linestyle='--',
    linewidth=1.5
)
#IP_der_ax.add_patch(circle)

OP_der_ax.invert_yaxis()

OP_der_ax.set_ylabel('Pial distance (\u03BCm)')
OP_der_ax.set_xlabel('Lateral spread (\u03BCm)')

IP_der_ax.set_ylabel('Pial distance (\u03BCm)')
IP_der_ax.set_xlabel('Lateral spread (\u03BCm)')

sns.ecdfplot(PV_csv,
             x='distance_from_mean_starter',
             hue='condition',
             palette=colors,
             ax=ecdf_pv_dist)
ecdf_pv_dist.legend_.set(title=None, frame_on=False)
ecdf_pv_dist.set_xlabel('PV distance to mean starter (\u03BCm)')



sns.barplot(PV_csv,
             y='distance_from_mean_starter',
             x='condition',
             hue='condition',
             palette=colors,
             errorbar='se',
             ax=bar_pv_dist,
             #alpha=.5
             )

bar_pv_dist.set_xlabel('')
bar_pv_dist.set_ylabel('PV distance to mean starter (\u03BCm)')
categories = PV_csv['condition'].unique()
test_stat,p, posthoc = pl.perform_stat_test(PV_csv, 'distance_from_mean_starter', 'condition')
pl.add_significance_bars(bar_pv_dist, posthoc, categories, y_offset=0.05)
print('PV distance from mean starter pos')
print(posthoc, test_stat)


locality_fraction = locality_fraction[locality_fraction.locality=='local']
sns.barplot(locality_fraction,
            y='fraction',
            x='condition',
            hue='condition',
            errorbar='se',
            palette=colors,
            ax=bar_locality,
            alpha=.5,
            order=list(colors.keys()),
            legend=False)
sns.stripplot(locality_fraction,
            y='fraction',
            x='condition',
            hue='condition',
            #dodge=True,
            palette=colors,
            order=list(colors.keys()),
            ax=bar_locality,
            legend=False)
bar_locality.set_xlabel('')
bar_locality.set_ylabel('Local PV inputs (%)')
categories = locality_fraction['condition'].unique()
test_stat,p, posthoc = pl.perform_stat_test(locality_fraction, 'fraction', 'condition')
print('locality')
print(posthoc, test_stat)
pl.add_significance_bars(bar_locality, posthoc, categories, y_offset=0.05)

# last row 
x_col = 'X_rel'
y_col = 'Y'
op = (PV_csv[PV_csv['condition'] == 'OP-der']
    .dropna(subset=[x_col, y_col])
    .copy())
ip = (PV_csv[PV_csv['condition'] == 'IP-der']
    .dropna(subset=[x_col, y_col])
    .copy())

OP_kde, OP_xy = f6_ut.generate_kde(op,x_col,y_col)
IP_kde, IP_xy = f6_ut.generate_kde(ip,x_col,y_col)

level = 0.8
OP_AR,_,_ = f6_ut.central_kde_aspect_ratio(OP_kde, OP_xy,level=level,
                                           show=True, 
                                           ax=example_ar_top, 
                                           color=colors['OP-der'],
                                           )
IP_AR, _,_ = f6_ut.central_kde_aspect_ratio(IP_kde, IP_xy,level=level,
                                            show=True,
                                            ax=example_ar_bottom, 
                                            color=colors['IP-der'],
                                            )

sns.histplot(data=simulated_data,
             x='aspect_ratio',
             ax=hist_ar,
             bins=50,
             stat='percent',
             color='grey')
observed = IP_AR
null = simulated_data.aspect_ratio.values
p_val = f6_ut.null_dist_p_test(observed,null,side='both')
hist_ar.axvline(
    observed,
    linestyle="--",
    label="Observed IP-der",
    color=colors['IP-der'],
)
hist_ar.axvline(
    OP_AR,
    linestyle="--",
    label="Observed OP-der",
    color=colors['OP-der'],
)
# 95% null boundaries
hist_ar.axvline(
    np.quantile(null, 0.975),
    linestyle=":",
    color='grey')
hist_ar.axvline(
    np.quantile(null, 0.025),
    linestyle=":",
    color='grey')
hist_ar.set_ylabel("Density")
#hist_ar.legend()
hist_ar.set_xlabel('Aspect ratio')
hist_ar.text(0.4,1.01,f'{pl.p_to_stars(p_val)}',
               fontsize=20,
               transform=hist_ar.transAxes,
               ha='center',
               va='center')
hist_ar.text(0.4,1.01,f'p = {p_val:.2g}',
               fontsize=10,
               transform=hist_ar.transAxes,
               ha='left',
               va='bottom')

op_modality_score = f6_ut.kde_multimodality_score(OP_kde,show=True,
                                                  ax=example_mode_top,
                                                  color=colors['OP-der'],
                                                  )
ip_modality_score = f6_ut.kde_multimodality_score(IP_kde,show=True,
                                                  ax=example_mode_bottom,
                                                  color=colors['IP-der'],
                                                  )

sns.histplot(data=simulated_data,
             x='modality_score',
             color='grey',
             bins=50,
             stat='percent',
             ax=hist_mode)
hist_mode.set_ylim(0,10)
observed = ip_modality_score
null = simulated_data.modality_score.values
p_val = f6_ut.null_dist_p_test(observed,null,side='both')
hist_mode.axvline(
    observed,
    linestyle="--",
    label="Observed \nIP-der",
    color=colors['IP-der'],
)
hist_mode.axvline(
    op_modality_score,
    linestyle="--",
    label="Observed \nOP-der",
    color=colors['OP-der'],
)
# 95% null boundaries
hist_mode.axvline(
    np.quantile(null, 0.975),
    linestyle=":",
    label=f"95% confidence \nintervals",
    color='grey')
hist_mode.axvline(
    np.quantile(null, 0.025),
    linestyle=":",
    color='grey')
hist_mode.set_ylabel("Density")
hist_mode.set_xlabel('Modality score')

hist_mode.text(0.4,1.01,f'{pl.p_to_stars(p_val)}',
               fontsize=20,
               transform=hist_mode.transAxes,
               ha='center',
               va='center')
hist_mode.text(0.4,1.01,f'p = {p_val:.2g}',
               fontsize=10,
               transform=hist_mode.transAxes,
               ha='left',
               va='bottom')
hist_mode.legend(
    loc="upper right",
    bbox_to_anchor=(1.6, 1),
    ncol=1,
    frameon=False
)

for ax in [hist_ar, example_mode_bottom, example_mode_top]:
    pos = ax.get_position()
    ax.set_position([
        pos.x0 - 0.04,  # Move left
        pos.y0,
        pos.width,
        pos.height
    ])
     # Move associated colorbars
    for cb in fig.axes:
        if cb.get_label() == "<colorbar>":
            cb_pos = cb.get_position()
            cb.set_position([
                cb_pos.x0 - 0.02,
                cb_pos.y0,
                cb_pos.width,
                cb_pos.height
            ])

sns.despine(fig,top=True,right=True)

fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/figure_6.svg',format='svg',bbox_inches=None)
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/figure_6.png')

plt.show()


