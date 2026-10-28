import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from code_utils import analysis_plotting as pl


plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

def Figure_S6(datapath,
              savepath
              ):


    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-OP': '#AE7DF8',
                    }
        
    palette_points = {'unlabeled': '#808080',
                    'IP-OP': '#AE7DF8',
                    }
    
    order = ['unlabeled','IP-OP']


    data_to_plot = pd.read_csv(datapath)
    data_to_plot["shared_input_amp"] = data_to_plot[["Neuron1_amplitude", "Neuron2_amplitude"]].mean(axis=1) 
    data_to_plot["Pair_dist"] = data_to_plot[["Neuron1_pial_dist", "Neuron2_pial_dist"]].mean(axis=1)
    data_to_plot = data_to_plot.groupby(['Pair_ID','Pair_type']).mean(numeric_only=True).reset_index()


    describe = data_to_plot.groupby(['Pair_type'])[['shared_input_amp','Pair_dist']].describe()
    print(describe)

    fig = plt.figure(figsize=(9.27, 11.69/4))

    gs = fig.add_gridspec(1, 2)
    shared_input_ax = fig.add_subplot(gs[0, 0])
    pial_dist_ax = fig.add_subplot(gs[0, 1])

    pl.add_panel_letter(fig, gs[0, 0], "a")
    pl.add_panel_letter(fig, gs[0, 1], "b")

    sns.despine(fig,right=True,top=True)

    y = 'shared_input_amp'
    shared_input_ax.set_xlabel(' ')
    shared_input_ax.set_ylabel('Shared input amplitude (pA)')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=shared_input_ax,
                order = order,
                alpha=.5,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=shared_input_ax)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'Pair_type')
    print(test_stat)
    print(posthoc)
    pl.add_significance_bars(shared_input_ax, posthoc, data_to_plot.Pair_type.unique(), y_offset=0.05)




    y = 'Pair_dist'
    pial_dist_ax.set_xlabel(' ')
    pial_dist_ax.set_ylabel('Pial distance (\u03BCm) ')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=pial_dist_ax,
                order = order,
                alpha=.5,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=pial_dist_ax)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'Pair_type')
    print(test_stat)
    print(posthoc)
    pl.add_significance_bars(shared_input_ax, posthoc, data_to_plot.Pair_type.unique(), y_offset=0.05)

    

    plt.show()

    plt.tight_layout()
    fig.savefig(os.path.join(savepath,'Fig_S6.svg'))
    fig.savefig(os.path.join(savepath,'Fig_S6.tiff'),dpi=600)
    

if __name__ == "__main__":


    Figure_S6(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 5/Fig_5_data.csv',
                savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 5/')
