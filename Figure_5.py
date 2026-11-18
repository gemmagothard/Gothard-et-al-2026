import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from code_utils import analysis_plotting as pl
import statsmodels.formula.api as smf

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

def Figure5(
        datapath,
        savepath,
        ):
    
    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-OP': '#AE7DF8'}
        
    palette_points = {'unlabeled': '#808080',
                    'IP-OP': '#AE7DF8'}
    
    red_green_palette = ['#00CDE5','#E404FF']

    data_to_plot = pd.read_csv(datapath)

    fig = plt.figure(figsize=(9.5, 6.69))

    # Main grid: 2 rows
    gs = fig.add_gridspec(2, 1,hspace=0.3)
    gs_top = gs[0].subgridspec(1, 3,wspace=0.5)

    empty1 = fig.add_subplot(gs_top[0, 0])
    empty2 = fig.add_subplot(gs_top[0, 1])
    empty3 = fig.add_subplot(gs_top[0, 2])

    for this_gs,ax, letter in zip(
        [gs_top[0, 0], gs_top[0, 1], gs_top[0, 2]],
        [empty1,empty2,empty3],
        ["a", "b", "c"]):
        ax.axis("off")
        pl.add_panel_letter(fig, this_gs, letter)


    gs_bottom = gs[1].subgridspec(1, 4,wspace=0.5)

    ax_outputprob = fig.add_subplot(gs_bottom[0, 0])
    ax_ampdiff   = fig.add_subplot(gs_bottom[0, 1])
    ax_dist      = fig.add_subplot(gs_bottom[0, 2])
    ax_pial      = fig.add_subplot(gs_bottom[0, 3])

    pl.add_panel_letter(fig,gs_bottom[0, 0], 'd')
    pl.add_panel_letter(fig,gs_bottom[0, 1], 'e')
    pl.add_panel_letter(fig,gs_bottom[0, 2], 'f')
    pl.add_panel_letter(fig,gs_bottom[0, 3],'g')

    sns.despine(fig,right=True,top=True)
    order = ['unlabeled', 'IP-OP']


    y = 'Shared_output_prob'
    ax_outputprob.set_xlabel(' ')
    ax_outputprob.set_ylabel('Shared output probability ')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=ax_outputprob,
                order = order,
                errorbar='se',
                edgecolor='black',
                alpha=.5)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=ax_outputprob)
    category='Pair_type'
    test_stat, p, posthoc = pl.perform_stat_test(data_to_plot,y,category)
    categories = data_to_plot[category].unique()
    print(test_stat,posthoc)
    pl.add_significance_bars(ax_outputprob, posthoc, categories,y_offset=0.1)
    

    y = 'Shared_amp_difference'
    ax_ampdiff.set_xlabel(' ')
    ax_ampdiff.set_ylabel('IPSC amplitude difference (%)')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=ax_ampdiff,
                order = order,
                errorbar='se',
                alpha=.5,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=ax_ampdiff)
    category='Pair_type'
    test_stat, p, posthoc = pl.perform_stat_test(data_to_plot,y,category)
    categories = data_to_plot[category].unique()
    print(test_stat,posthoc)
    pl.add_significance_bars(ax_ampdiff, posthoc, categories, y_offset=0.1)
    

    y = 'Distance_apart'
    ax_dist.set_xlabel(' ')
    ax_dist.set_ylabel('Inter-neuronal distance (\u03BCm)')
    data_pairs = data_to_plot.groupby(['Pair_ID','Pair_type'])[['Distance_apart',
                                                                'Neuron1_pial_dist',
                                                                'Neuron2_pial_dist',
                                                                'Neuron1_ID',
                                                                'Neuron2_ID']].first().reset_index()
    sns.barplot(data=data_pairs,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=ax_dist,
                errorbar='se',
                alpha=.5,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_pairs,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=ax_dist)
    

    OP_derived = np.concat([data_pairs[data_pairs.Neuron1_ID=='OP-derived'].Neuron1_pial_dist,
                    data_pairs[data_pairs.Neuron2_ID=='OP-derived'].Neuron2_pial_dist])
    IP_derived = np.concat([data_pairs[data_pairs.Neuron2_ID=='IP-derived'].Neuron2_pial_dist,
                data_pairs[data_pairs.Neuron1_ID=='IP-derived'].Neuron1_pial_dist])
    data = [OP_derived, IP_derived]
    pl.scatter_bar_plot(ax_pial, data, red_green_palette, ['OP-der','IP-der'], True ,'Pial distance (\u03BCm)', False, savepath,False)

    
    fig.savefig(os.path.join(savepath,'Fig_5.svg'))
    plt.show()


if __name__ == "__main__":

    Figure5(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 5/Fig_5_data.csv',
            savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 5/')

