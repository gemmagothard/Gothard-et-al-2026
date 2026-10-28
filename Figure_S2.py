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


def FigureS2(datapath,
             savepath):
    
    plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
    })


    data = pd.read_csv(datapath)

    palette_bar = { 'Ta1+ Glast-'   : '#EAF3E4',
                    'Ta1+ Glast+'   : '#FDF1DB',
                    'Ta1- Glast+'   : '#F4D1D1'}

    palette_points = { 'Ta1+ Glast-'   : '#8DC170',
                        'Ta1+ Glast+'   : '#EC8F0E',
                        'Ta1- Glast+'   : '#C40D0D'}  
    
    order = ['Ta1+ Glast-', 'Ta1+ Glast+', 'Ta1- Glast+']
    order_pial_dist = ['Ta1+ Glast-','Ta1- Glast+']

    fig = plt.figure(figsize=(7.27, (11.69/4)*2))
    gs = fig.add_gridspec(2, 3, wspace=0.3, height_ratios=[1,1.2])

    pl.add_panel_letter(fig, gs[0, 0], "a")
    pl.add_panel_letter(fig, gs[0, 2], "b")

    pl.add_panel_letter(fig, gs[1, 0], "c")
    pl.add_panel_letter(fig, gs[1, 2], "d")

    cell_type_ax = fig.add_subplot(gs[1,0])
    venn_ax = fig.add_subplot(gs[1,1])
    pial_dist_ax = fig.add_subplot(gs[1,2])

    sns.barplot(data=data,
                x='cell_id',
                y='cell_counts',
                ax=cell_type_ax,
                order=order,
                hue='cell_id',
                errorbar='se',
                palette=palette_bar)
    sns.stripplot(data=data,
            x='cell_id',
            y='cell_counts',
            ax=cell_type_ax,
            order=order,
            hue='cell_id',
            palette=palette_points)
    cell_type_ax.set_ylabel('L2/3 cell type (%)')
    cell_type_ax.set_xlabel('')
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order)}
    for pid, group in data.groupby('Brain'):
        available = [label for label in order if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['cell_counts'].values
            cell_type_ax.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)



    data_pial_dist = data[data.cell_id != 'Ta1+ Glast+']
    sns.barplot(data=data_pial_dist,
                x='cell_id',
                y='pial_dist',
                ax=pial_dist_ax,
                order=order_pial_dist,
                hue='cell_id',
                errorbar='se',
                palette=palette_bar)
    sns.stripplot(data=data_pial_dist,
            x='cell_id',
            y='pial_dist',
            ax=pial_dist_ax,
            order=order_pial_dist,
            hue='cell_id',
            palette=palette_points)
    pial_dist_ax.set_ylabel('Pial distance (\u03BCm)')
    pial_dist_ax.set_xlabel('')

    
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order_pial_dist)}
    for pid, group in data_pial_dist.groupby('Brain'):
        available = [label for label in order_pial_dist if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['pial_dist'].values
            pial_dist_ax.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)

    # Plot Venn diagram with green and yellow only
    size_green = np.round(np.average(data[data.cell_id=='Ta1+ Glast-'].cell_counts),0)
    size_yellow = np.round(np.average(data[data.cell_id=='Ta1+ Glast+'].cell_counts),0)
    size_red = np.round(np.average(data[data.cell_id=='Ta1- Glast+'].cell_counts),0)

    venn2(
        subsets=(size_green, size_red, size_yellow),
        set_colors=(palette_points['Ta1+ Glast-'], palette_points['Ta1- Glast+']),
        alpha=0.8,
        ax=venn_ax
    )

    sns.despine(fig,right=True,top=True)
    fig.savefig(os.path.join(savepath, 'Figure_S2.svg'))

    plt.show()
    



FigureS2(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1/Fig_S2_data.csv',
         savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1')