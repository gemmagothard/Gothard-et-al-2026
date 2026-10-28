import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from matplotlib_venn import venn2
from matplotlib.patches import Patch

from code_utils import analysis_plotting as pl

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})



def FigureS1(datapath,
             savepath):
    

    data = pd.read_csv(datapath)

    fig = plt.figure(figsize=(9.27, 10.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6,1,1])

    gs0 = gs[0].subgridspec(1, 2, width_ratios=[1, 4])
    pl.add_panel_letter(fig, gs0[0, 0], "a")
    pl.add_panel_letter(fig, gs0[0, 1], "b")

    gs1 = gs[1].subgridspec(1, 2, width_ratios=[1, 2])
    pl.add_panel_letter(fig, gs1[0, 0], "c")
    pl.add_panel_letter(fig, gs1[0, 1], "d")

    gs2 = gs[2].subgridspec(1, 2, width_ratios=[1, 0.7])
    pl.add_panel_letter(fig, gs2[0, 0], "e")
    pl.add_panel_letter(fig, gs2[0, 1], "f")
    

    gs_sub = gs2[0,1].subgridspec(2,2,wspace=0.9,height_ratios=[0.5,1])
    pl.add_panel_letter(fig, gs_sub[0,1], "g")
    ax_venn = fig.add_subplot(gs_sub[0,0])
    ax_bar = fig.add_subplot(gs_sub[1,0])
    
    ax_legend = fig.add_subplot(gs_sub[0,1])
    ax_violin = fig.add_subplot(gs_sub[1,1])


    palette = { 'GFP+'      : '#8DC170',
                'GFP+TdT+'  : '#EF8706',
                'red'       :'#C70909'} 

    label_map = {'GFP+TdT+' :'basal IP-derived',
                'GFP+'      :'apical IP-derived'}

    order = ['GFP+','GFP+TdT+']

    ax_legend.axis('off')

    # Create custom legend handles in desired order
    legend_elements = [
        Patch(facecolor=palette[cell], edgecolor='none', label=label_map[cell])
        for cell in ['GFP+', 'GFP+TdT+']
    ]
    # Draw the legend in the custom subplot
    ax_legend.legend(
        handles=legend_elements,
        loc='center',
        frameon=False,
        fontsize='medium'
    )

    ## new bar plot with only green / yellow cells
    sns.barplot(data=data,
                y='counts_normalized',
                x='cell_id',
                hue='cell_id',
                estimator='mean',
                errorbar='se',
                palette=palette,
                alpha=0.5,
                ax=ax_bar,
                order=order,
                )
    sns.stripplot(data=data,
                y='counts_normalized',
                x='cell_id',
                hue='cell_id',
                order=order,
                ax=ax_bar,
                palette=palette,
                jitter=0,
                alpha=1,
                )
    ax_bar.set_ylabel('L2/3 cell type (%)')
    ax_bar.set_xlabel('')
    ax_bar.xaxis.set_visible(False)
    # Optional: connect paired points with lines
    # Assumes a 'pair_id' column exists indicating matched samples
    pair_col = 'experiment_ID'  # <-- change this to your actual column name
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order)}
    for pid, group in data.groupby('experiment_ID'):
        available = [label for label in order if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['counts_normalized'].values
            ax_bar.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)


    sns.violinplot(data=data,
                y='pial_dist',
                x='cell_id',
                hue='cell_id',
                ax=ax_violin,
                palette=palette,
                inner='box',  # 'box', 'quartile', 'point', 'stick', or None
                split=True,
                alpha=1
                )
    # sns.pointplot(data=data,
    #             y='pial_dist',
    #             x='cell_id',
    #             ax=ax_violin,
    #             alpha=1,
    #             linestyle="none",
    #             palette=['black','black'],
    #             )
    ax_violin.invert_yaxis()
    ax_violin.set_ylabel('Pial distance (\u03bcm)')
    ax_violin.xaxis.set_visible(False)

    # Plot Venn diagram with green and yellow only
    size_green = np.round(np.nanmean(data[data.cell_id=='GFP+'].counts_normalized),0)
    size_yellow = np.round(np.nanmean(data[data.cell_id=='GFP+TdT+'].counts_normalized),0)
    size_red = 0

    venn2(
        subsets=(size_green, size_red, size_yellow),
        set_colors=(palette['GFP+'], palette['red']),
        alpha=0.8,
        ax=ax_venn
        )

    sns.despine(fig, ['top', 'right'])
    fig.savefig(os.path.join(savepath, 'Figure_S1.svg'))

    plt.show()



FigureS1(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1/Fig_S1_data.csv',
         savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1'
         )