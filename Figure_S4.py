
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from code_utils import analysis_plotting as pl
from pathlib import Path 

def plot_regression_stats(data, x, y, ax):
    """
    Plot linear regression with R² and p-value.
    """
    from scipy.stats import linregress
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Remove NaNs
    df = data[[x, y]].dropna()

    # Regression
    slope, intercept, r, p, stderr = linregress(df[x], df[y])
    r2 = r**2

    # Plot
    sns.regplot(
        data=df,
        x=x,
        y=y,
        ax=ax
    )

    # Add statistics
    ax.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        va="top"
    )


colors_neurons = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }

colors_pairs = {
                'OP-OP':'#00CDE5',
                'IP-OP': '#AE7DF8',
                'IP-IP':'#E404FF',
                    }

path  = Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 2/')

ipsc_shared_results = pd.read_csv(path / 'ipsc_results.csv')
ipsc_results_by_neuron = pd.read_csv(path / 'ipsc_results_by_neuron_2026.csv')

epsc_shared_results = pd.read_csv(path / 'epsc_results.csv')
epsc_results_by_neuron = pd.read_csv(path / 'epsc_results_by_neuron_2026.csv')

ipsc_shared_results = ipsc_shared_results.dropna(subset=['percent_shared_mean'])
epsc_shared_results = epsc_shared_results.dropna(subset=['percent_shared_mean'])

ipsc_results_by_neuron = ipsc_results_by_neuron.dropna(subset=['percent_shared'])
epsc_results_by_neuron = epsc_results_by_neuron.dropna(subset=['percent_shared'])

print(f'IPSC pairs N: \n {ipsc_shared_results.groupby(['pair_type'])['pair_id'].count()}')
print(f'IPSC pairs, n of mice: \n {ipsc_shared_results.groupby(['pair_type'])['mouse_id'].nunique()}')

print(f'EPSC pairs, N: \n {epsc_shared_results.groupby(['pair_type'])['pair_id'].count()}')
print(f'EPSC pairs. n of mice: \n {epsc_shared_results.groupby(['pair_type'])['mouse_id'].nunique()}')

print(f'IPSC, n of neurons: \n {ipsc_results_by_neuron.groupby(['neuron_type'])['freq'].count()}')
print(f'IPSC, n of mice: \n {ipsc_results_by_neuron.groupby(['neuron_type'])['mouse_id'].nunique()}')

print(f'EPSC, n of neurons: \n {epsc_results_by_neuron.groupby(['neuron_type'])['freq'].count()}')
print(f'EPSC, n of mice: \n {epsc_results_by_neuron.groupby(['neuron_type'])['mouse_id'].nunique()}')

fig = plt.figure(figsize=(9.27, 6.69))

gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    wspace=0.5,
    right=0.95
)

ax_ipsc_distance  = fig.add_subplot(gs[1, 1])
ax_ipsc_pial_dist = fig.add_subplot(gs[1, 2])
ax_ipsc_freq      = fig.add_subplot(gs[0, 1])
ax_ipsc_amp       = fig.add_subplot(gs[0, 0])

#ax_epsc_distance  = fig.add_subplot(gs[2, 0])
#ax_epsc_pial_dist = fig.add_subplot(gs[2, 1])
ax_epsc_freq      = fig.add_subplot(gs[1, 0])
ax_epsc_amp       = fig.add_subplot(gs[0, 2])

pl.add_panel_letter(fig, gs[0, 0], 'a')
pl.add_panel_letter(fig, gs[0, 1], 'b')
pl.add_panel_letter(fig, gs[0, 2], 'c')

pl.add_panel_letter(fig, gs[1, 0], 'd')
pl.add_panel_letter(fig, gs[1, 1], 'e')
pl.add_panel_letter(fig, gs[1, 2], 'f')

dv = 'pipette_dist_mean'
iv = 'pair_type'
sns.barplot(ipsc_shared_results,
            x=iv,
            y=dv,
            hue=iv,
            errorbar='se',
            palette=colors_pairs,
            ax=ax_ipsc_pial_dist,
            order=list(colors_pairs.keys()),
            alpha=.5,
            )
sns.stripplot(ipsc_shared_results,
            x=iv,
            y=dv,
            hue=iv,
            palette=colors_pairs,
            ax=ax_ipsc_pial_dist
            )
ax_ipsc_pial_dist.set_ylabel('Pial distance (µm)')

test_stat, p, posthoc = pl.perform_stat_test(ipsc_shared_results, 
                                                dv, 
                                                iv)
print('ipsc', dv, f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_ipsc_pial_dist, posthoc, ipsc_shared_results[iv].unique().tolist(), y_offset=0.6)

# dv = 'pipette_dist_mean'
# iv = 'pair_type'
# sns.barplot(epsc_shared_results,
#             x=iv,
#             y=dv,
#             hue=iv,
#             errorbar='se',
#             palette=colors_pairs,
#             order=list(colors_pairs.keys()),
#             ax=ax_epsc_pial_dist,
#             alpha=.5
#             )
# sns.stripplot(epsc_shared_results,
#             x=iv,
#             y=dv,
#             hue=iv,
#             palette=colors_pairs,
#             ax=ax_epsc_pial_dist
#             )
# ax_epsc_pial_dist.set_ylabel('Pial distance (µm)')
# print('epsc',dv)
# test_stat, p, posthoc = pl.perform_stat_test(epsc_shared_results, 
#                                                 dv, 
#                                                 iv)
# if posthoc is not np.nan:
#     pl.add_significance_bars(ax_epsc_pial_dist, posthoc, epsc_shared_results[iv].unique().tolist(), y_offset=0.1)


dv = 'distance_apart'
iv = 'pair_type'
sns.barplot(ipsc_shared_results,
            x=iv,
            y=dv,
            hue=iv,
            errorbar='se',
            palette=colors_pairs,
            order=list(colors_pairs.keys()),
            ax=ax_ipsc_distance,
            alpha=.5
            )
sns.stripplot(ipsc_shared_results,
            x=iv,
            y=dv,
            hue=iv,
            palette=colors_pairs,
            ax=ax_ipsc_distance
            )
ax_ipsc_distance.set_ylabel('Inter-neuronal distance (µm)')
test_stat, p, posthoc = pl.perform_stat_test(ipsc_shared_results, 
                                                dv, 
                                                iv)
print('ipsc',dv,f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_ipsc_distance, posthoc, ipsc_shared_results[iv].unique().tolist(), y_offset=0.6)


# sns.barplot(epsc_shared_results,
#             x=iv,
#             y=dv,
#             hue=iv,
#             errorbar='se',
#             palette=colors_pairs,
#             order=list(colors_pairs.keys()),
#             ax=ax_epsc_distance,
#             alpha=.5
#             )
# sns.stripplot(epsc_shared_results,
#             x=iv,
#             y=dv,
#             hue=iv,
#             palette=colors_pairs,
#             ax=ax_epsc_distance
#             )
# ax_epsc_distance.set_ylabel('Inter-neuronal distance (µm)')
# print('epsc',dv)
# test_stat, p, posthoc = pl.perform_stat_test(epsc_shared_results, 
#                                                 dv, 
#                                                 iv)
# if posthoc is not np.nan:
#     pl.add_significance_bars(ax_epsc_distance, posthoc, epsc_shared_results[iv].unique().tolist(), y_offset=0.6)

dv = 'freq'
iv = 'neuron_type'
sns.barplot(ipsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            order = list(colors_neurons.keys()),
            errorbar='se',
            palette=colors_neurons,
            ax=ax_ipsc_freq,
            alpha=.5,
            )
sns.stripplot(ipsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            order = list(colors_neurons.keys()),
            palette=colors_neurons,
            ax=ax_ipsc_freq
            )
ax_ipsc_freq.set_ylabel('sIPSC frequency (Hz)')
test_stat, p, posthoc = pl.perform_stat_test(ipsc_results_by_neuron, 
                                                dv, 
                                                iv)
print('ipsc',dv,f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_ipsc_freq, posthoc, ipsc_results_by_neuron[iv].unique().tolist(), y_offset=0.1)

sns.barplot(epsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            errorbar='se',
            palette=colors_neurons,
            order = list(colors_neurons.keys()),
            ax=ax_epsc_freq,
            alpha=.5
            )
sns.stripplot(epsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            palette=colors_neurons,
            order = list(colors_neurons.keys()),
            ax=ax_epsc_freq
            )
ax_epsc_freq.set_ylabel('sEPSC frequency (Hz)')
test_stat, p, posthoc = pl.perform_stat_test(epsc_results_by_neuron, 
                                                dv, 
                                                iv)
print('epsc',dv,f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_epsc_freq, posthoc, epsc_results_by_neuron[iv].unique().tolist(), y_offset=0.6)


dv = 'amp'
sns.barplot(ipsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            errorbar='se',
            palette=colors_neurons,
            ax=ax_ipsc_amp,
            order = list(colors_neurons.keys()),
            alpha=.5
            )
sns.stripplot(ipsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            palette=colors_neurons,
            ax=ax_ipsc_amp,
            order = list(colors_neurons.keys()),
            )
ax_ipsc_amp.set_ylabel('sIPSC amplitude (pA)')
test_stat, p, posthoc = pl.perform_stat_test(ipsc_results_by_neuron, 
                                                dv, 
                                                iv)
print('ipsc',dv,f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_ipsc_amp, posthoc, ipsc_results_by_neuron[iv].unique().tolist(), y_offset=0.6)


sns.barplot(epsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            errorbar='se',
            palette=colors_neurons,
            ax=ax_epsc_amp,
            order = list(colors_neurons.keys()),
            alpha=.5
            )
sns.stripplot(epsc_results_by_neuron,
            x=iv,
            y=dv,
            hue=iv,
            palette=colors_neurons,
            ax=ax_epsc_amp,
            order = list(colors_neurons.keys()),
            )
ax_epsc_amp.set_ylabel('sEPSC amplitude (pA)')
test_stat, p, posthoc = pl.perform_stat_test(epsc_results_by_neuron, 
                                                dv, 
                                                iv)
print('epsc',dv,f'T = {test_stat:3g}, P = {p:.3g}')
if posthoc is not np.nan:
    pl.add_significance_bars(ax_epsc_amp, posthoc, epsc_results_by_neuron[iv].unique().tolist(), y_offset=0.6)


for ax in fig.axes:
    ax.set_xlabel('')

sns.despine(fig, top=True, right=True)
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 2/Figure_S4.svg',dpi=600)
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 2/Figure_S4.tiff',dpi=600)

plt.show()

