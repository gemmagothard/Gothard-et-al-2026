
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from code_utils import analysis_plotting as pl
from add_example_traces import add_example_to_plot

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})



def GenerateFigure(
        datapath,
        ephys_datapath,
        savepath,
        ):
    
    fig = plt.figure(figsize=(8.27, 11.69/4.5))

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.5,2,1],
        wspace=0.5
    )
    pl.add_panel_letter(fig, gs[0, 0], 'a')
    pl.add_panel_letter(fig, gs[0, 2], 'b')

    ax = fig.add_subplot(gs[0, 2])
    ax_example = fig.add_subplot(gs[0,1])
    ax_example.axis('off')
    add_example_to_plot(fig, gs[0,1], example='unlabeled',type='ipsc')

    unlabeled_data = pd.read_csv(ephys_datapath)
    unlabeled_data = unlabeled_data[unlabeled_data.pair_type=='unlabeled']
    mean = unlabeled_data.percent_shared_mean.mean()
    sem = unlabeled_data.percent_shared_mean.sem()
    print(f'Mean: {mean:.3f}')
    print(f'SEM: {sem:.3f}')
    sns.barplot(unlabeled_data,
                x='pair_type',
                y='percent_shared_mean',
                color='grey',
                alpha=0.5,
                ax=ax,
                errorbar='se')
    sns.stripplot(unlabeled_data,
                x='pair_type',
                y='percent_shared_mean',
                color='grey',
                ax=ax,
                )
    ax.set_ylabel('Shared inhibitory inputs (%)')
    ax.set_xlabel(' ')
    ax.set_xticklabels([])
    ax.set_xticks([])
    sns.despine(fig,top=True,right=True)

    fig.savefig(savepath / 'Figure_S5.svg')
    
    plt.show()

    biases_data = pd.read_csv(datapath / 'simulations' / 'simulate_network_empirical_biases.csv')
    biases_data = biases_data[
            (biases_data['Condition'] == 'Testing')
        ]
    biases_data['Bias_type'] = 'Biased inhibition'

    unbiased_data = pd.read_csv(datapath / 'simulations' / 'simulate_network_unbiased.csv')
    unbiased_data = unbiased_data[
            (unbiased_data['Condition'] == 'Testing')
        ]
    unbiased_data['Bias_type'] = 'Unbiased inhibition'
    
    data = pd.concat([biases_data,unbiased_data])

    sns.barplot(data,
                hue='Population',
                y='Firing rate [Hz]',
                x='Bias_type',
                errorbar='se')
    plt.show()
    

GenerateFigure(
        datapath=Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 3/data_20261112'),
        ephys_datapath=Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 2/ipsc_results.csv'),
        savepath=Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 3') 
)