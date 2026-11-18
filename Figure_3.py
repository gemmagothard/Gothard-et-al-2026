
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import json
from pathlib import Path
from code_utils import analysis_plotting as pl
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

def smooth(values, dt=50):
    from scipy.signal import convolve
    from scipy.signal.windows import gaussian
    window = gaussian(6 * dt, dt)
    window /= window.sum()
    return convolve(values, window, mode="same")

def plot_spikes(times, indices, ax, color, n=70):
    unique_indices = np.unique(indices)
    mask = np.isin(indices, unique_indices[:n])
    indices, times = indices[mask], times[mask]
    ax.plot(times, indices, ".", color=color, markersize=1)

def get_firing_rate(times_in_ms, dt=50, minlength=0):
        spikes_per_ms = np.bincount(np.array(times_in_ms, dtype=int), minlength=minlength)
        return smooth(spikes_per_ms, dt=dt)

def Figure3(
        datapath,
        savepath,
        ):
        
    input_file = "295b00d0-b0ef-488d-9fb8-a6dd65942461"
    cross_corr = pd.read_csv(datapath / 'simulations' / 'cross_correlations.csv')
    inputs = np.load(datapath / 'inputs' / str(input_file + '.npz'), allow_pickle=True)
    input_magnitudes = pd.read_json(datapath / 'inputs/input_magnitudes.json')

    
    biases_data = pd.read_csv(datapath / 'simulations' / 'simulate_network_empirical_biases.csv')
    biases_data = biases_data[
            (biases_data['Condition'] == 'Testing') &
            (biases_data['Inputs'].str.contains(input_file, na=False, regex=False))
        ]
    unbiased_data = pd.read_csv(datapath / 'simulations' / 'simulate_network_unbiased.csv')
    unbiased_data = unbiased_data[
            (unbiased_data['Condition'] == 'Testing') &
            (unbiased_data['Inputs'].str.contains(input_file, na=False, regex=False))
        ]
    bias_data = [biases_data,unbiased_data]

    population_size = dict()
    population_size["IP-derived pyramidal neurons"] = 500
    population_size["OP-derived pyramidal neurons"] = 500
    population_size["IP-targeting interneurons"]    = 125
    population_size["OP-targeting interneurons"]    = 125

    pyramidal_neurons = ['IP-derived pyramidal neurons','OP-derived pyramidal neurons']
    pyr_to_color = {'IP-derived pyramidal neurons':'#E404FF',
                    'OP-derived pyramidal neurons':'#00CDE5'}
    palette_population = {'Unbiased':'lightgrey',
                        'Biased':'darkgrey'}
    label_dict = {'IP-derived pyramidal neurons':'IP-der \nspike times',
                'OP-derived pyramidal neurons':'OP-der \nspike times',
                }

    cross_corr = cross_corr[cross_corr.Source.isin(pyramidal_neurons) & cross_corr.Target.isin(pyramidal_neurons)]
    zero_lag = cross_corr[cross_corr['Lag [ms]'] == 0]

    correlation_population = zero_lag.groupby('Connectivity bias')['Correlation coefficient'].describe()
    correlation_population['sem'] = (
        correlation_population['std'] / np.sqrt(correlation_population['count'])
    )
    print(correlation_population[['mean','sem']])

    fig = plt.figure(figsize=(8.27, 11))
    gs = fig.add_gridspec(
        nrows=7,
        ncols=2,
        height_ratios=[2.2, 1, 0.7, 0.7, 1, 0.3, 2.5],
        wspace=0.3,
        hspace=0.5
    )

    # Create 12 axes
    axs = [[fig.add_subplot(gs[i, j]) for j in range(2)]
        for i in range(7) if i != 5]

    axs = np.array(axs)
    axs[0,0].axis('off')
    offset = -0.1
    axs[0,0].set_title('Biased inhibition')
    pl.add_panel_letter(fig, axs[0,0], 'a',x_offset=offset)
    axs[0,1].axis('off')
    axs[0,1].set_title('Unbiased inhibition')
    pl.add_panel_letter(fig, axs[0,1], 'b',x_offset=offset)

    pl.add_panel_letter(fig, axs[1,0], 'c', x_offset=offset)
    pl.add_panel_letter(fig, axs[1,1], 'd', x_offset=offset)

    pl.add_panel_letter(fig, axs[-1,-2], 'e',x_offset=offset)
    pl.add_panel_letter(fig, axs[-1,-1], 'f',x_offset=offset)


    # First column (biased)
    axs_bias = axs[1:, 0]
    # Second column (unbiased)
    axs_unbias = axs[1:, 1]

    handles = [
        Line2D([0], [0], color=pyr_to_color[pyr], lw=1, label=pyr)
        for pyr in pyr_to_color
    ]
    axs_unbias[0].legend(
        handles=handles,
        loc='lower left',
        ncol=1,
        bbox_to_anchor=(0.4, 0.85),
        frameon=False,
        handlelength=0.8,
        fontsize=8
    )

    # loop round biased or unbiased data
    for bias, ax_column in zip(bias_data,[axs_bias,axs_unbias]):

        # plot stim trace
        for pyr in pyramidal_neurons:
            stop = bias[bias.Population==pyr]['Stop [ms]'].values
            start= stop-2000

            # plot stimulus input
            x = np.arange(start, stop)
            y = smooth(inputs[f"{pyr} (population)"] * input_magnitudes.loc["population",pyr],dt=10)[int(start):int(stop)]
            ax_column[0].plot(x, y, c=pyr_to_color[pyr],linewidth=1)
            ax_column[0].plot(x, y, c=pyr_to_color[pyr],linewidth=1)
            ax_column[0].set_ylabel('Input \nmagnitude (pA)')

        # plot spike times
        for pyr,ax in zip(pyramidal_neurons,[ax_column[1],ax_column[2]]):
            stop = bias[bias.Population==pyr]['Stop [ms]'].values
            start= stop-2000
            uid = bias.loc[bias.Population == pyr, 'uid'].iloc[0]
            spikes = np.load(datapath / 'simulations' / 'spikes' / f'{uid}.npz', allow_pickle=True)
            indices, times = spikes[pyr].T
            valid = np.logical_and(times >= start, times<stop)
            indices, times = indices[valid], times[valid]
            plot_spikes(times, indices, ax, pyr_to_color[pyr])
            ax.set_ylabel(f'{label_dict[pyr]}')
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.set_yticks([])
            
        # plot firing rates
        for pyr, ax in zip(pyramidal_neurons,[ax_column[3],ax_column[3]]):
            stop = int(bias[bias.Population==pyr]['Stop [ms]'].values)
            start= stop-2000
            dt = 10
            x = (np.arange(start, stop) - start) / 1000
            indices, times = spikes[pyr].T
            firing_rate = 1000 * get_firing_rate(times, dt=dt) / population_size[pyr]
            y = firing_rate[start:stop]
            ax.plot(x[:len(y)], y, color=pyr_to_color[pyr], linewidth=1, label=pyr) # when there are no spikes in the last millisecond, the firing rate array is short
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing \nrate (Hz)')
            ax.spines[['top','right']].set_visible(False)

    # remove ticks 
    for col in range(2):
        for ax in axs[:-2, col]:
            ax.set_xticklabels([])
            ax.spines[['bottom','top','right']].set_visible(False)
            ax.tick_params(axis='x', which='minor', bottom=False)
            ax.tick_params(axis='x', which='major', bottom=True)
            

    # population data
    sns.lineplot(cross_corr,
                x='Lag [ms]',
                y='Correlation coefficient',
                hue='Connectivity bias',
                estimator='mean',
                errorbar='se',
                palette=palette_population,
                ax=axs[-1][0]
                    )
    axs[-1][0].set_xlabel('Lag (ms)')

    sns.barplot(zero_lag,
                x='Connectivity bias',
                y='Correlation coefficient',
                hue='Connectivity bias',
                estimator='mean',
                errorbar='se',
                palette=palette_population,
                ax=axs[-1][1]
                    )
    iv = 'Connectivity bias'
    dv = 'Correlation coefficient'
    categories = zero_lag[iv].unique()
    test_stat,p, posthoc = pl.perform_stat_test(zero_lag, dv, iv)
    print(test_stat)
    print(p)
    print(posthoc)
    if posthoc is not np.nan:
        pl.add_significance_bars(axs[-1][1], posthoc, categories, y_offset=0.1)

    axs[-1][1].spines[['top','right']].set_visible(False)
    axs[-1][0].spines[['top','right']].set_visible(False)

    fig.align_ylabels(axs_bias)
    fig.align_ylabels(axs_unbias)
    fig.savefig(savepath / 'Figure_3.svg')

    plt.show()


Figure3(
        datapath=Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 3/data_20261112'),
        savepath=Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 3') 
)