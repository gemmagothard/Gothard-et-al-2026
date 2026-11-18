import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from code_utils import analysis_plotting as pl

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

palette_bars = {
        'Unlabeled':'grey',
        'IP-der': '#CB23E0',
        }

cell_type_map = {'IP-derived':'IP-der',
                 'Unlabeled':'Unlabeled'}

def Figure7(
        NBQX_datapath,
        traces_datapath,
        cell_datapath,
        savepath,
        ):

    order = ['Unlabeled','IP-der']
    
    NBQX_data = pd.read_csv(NBQX_datapath)
    traces = pd.read_csv(traces_datapath)
    cells = pd.read_csv(cell_datapath)
    #cells['cell_type'] = cells['cell_type'].map(cell_type_map)
    #traces['cell_type'] = traces['cell_type'].map(cell_type_map)

    cells = cells[np.logical_or(cells.cell_type=='IP-der',cells.cell_type=='Unlabeled')]
    traces = traces[traces.cell_type.isin(['IP-der','Unlabeled'])]


    described = cells.groupby(['cell_type'])[['IT_lfp','STTC']].describe()
    for col in ['IT_lfp', 'STTC']:
        described[(col, 'sem')] = described[(col, 'std')] / np.sqrt(described[(col, 'count')])
    print(described.loc[:, (slice(None), ['count', 'mean', 'sem'])])

    # MAKE FIGURE - A4 portrait
    fig = plt.figure(figsize=(9.27, 11.69))

    # Main grid: 4 rows
    gs = fig.add_gridspec(4, 1, hspace=0.4, height_ratios=[1,0.7,1,1])

    # Row 1: 3 panels
    gs0 = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 1])
    pl.add_panel_letter(fig, gs0[0, 0], "a")
    pl.add_panel_letter(fig, gs0[0, 1], "b")
    pl.add_panel_letter(fig, gs0[0, 2], "c")
    NBQX_plot = fig.add_subplot(gs0[0,2])

    # row 2:
    gs1 = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1])
    pl.add_panel_letter(fig, gs1[0, 0], "d")
    pl.add_panel_letter(fig, gs1[0, 1], "e")

    # row 3
    gs2 = gs[2].subgridspec(1, 3, width_ratios=[2,1,0.7])
    it_LFP_examples = fig.add_subplot(gs2[0,1])
    it_LFP_averages = fig.add_subplot(gs2[0,2],sharey=it_LFP_examples)
    pl.add_panel_letter(fig, gs2[0, 0], "f")
    pl.add_panel_letter(fig, gs2[0, 1], "g")

    # row 4
    gs3 = gs[3].subgridspec(1, 3, width_ratios=[2,1,0.7])
    ETTC_barplot = fig.add_subplot(gs3[0,1:])
    pl.add_panel_letter(fig, gs3[0, 0], "h")
    pl.add_panel_letter(fig, gs3[0, 1], "i")

        
    # IP_mean = np.mean(traces['IP-der'],axis=0)
    # IP_sem = st.sem(traces['IP-der'],axis=0)
    # unlabeled_mean = np.mean(traces['unlabeled'],axis=0)
    # unlabeled_sem = st.sem(traces['unlabeled'],axis=0)

    # it_LFP_examples.plot(traces['timevec_trace'],IP_mean,color=palette_bars['IP-der'])
    # it_LFP_examples.fill_between(traces['timevec_trace'],IP_mean+IP_sem,IP_mean-IP_sem,color=palette_bars['IP-der'],alpha=.2)
    # it_LFP_examples.plot(traces['timevec_trace'],unlabeled_mean,color=palette_bars['Unlabeled'])
    # it_LFP_examples.fill_between(traces['timevec_trace'],unlabeled_mean+unlabeled_sem,unlabeled_mean-unlabeled_sem,color=palette_bars['Unlabeled'],alpha=.2)
    sns.lineplot(traces,
             x='timevec',
             y='lfp_windows',
             hue='cell_type',
             palette=palette_bars,
             errorbar='se',
             estimator='mean',
             ax=it_LFP_examples,
             legend=False
             )
    it_LFP_examples.set_ylabel('IT-LFP (-\u03BCV)')
    it_LFP_examples.set_xlabel('Time lags (ms)')
    it_LFP_examples.axvline(0,linestyle='dotted',color='black')
    it_LFP_examples.set_xlim(-100,100)


    y = 'IT_lfp'
    this_ax = it_LFP_averages
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel(' ')
    sns.barplot(data=cells,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                alpha=.5,
                errorbar='se',
                edgecolor='black',)
    sns.stripplot(data=cells,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_bars,
                order = order,
                ax=this_ax)
    test_stat,p, posthoc = pl.perform_stat_test(cells, y, 'cell_type')
    print(test_stat,p,posthoc)
    pl.add_significance_bars(this_ax, posthoc, cells.cell_type.unique(), y_offset=0.05)

    
    y = 'STTC'
    this_ax = ETTC_barplot
    sns.barplot(data=cells,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                alpha=.5,
                ax=this_ax,
                order = order,
                errorbar='se',
                edgecolor='black',)
    sns.stripplot(data=cells,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_bars,
                order = order,
                ax=this_ax)
    this_ax.set_ylabel('ETTC')
    this_ax.set_xlabel(' ')
    test_stat,p, posthoc = pl.perform_stat_test(cells, y, 'cell_type')
    print(f'ETTC T = {test_stat:3g}, P = {p:3g}')
    pl.add_significance_bars(this_ax, posthoc, cells.cell_type.unique(), y_offset=0.05)

    NBQX_describe = NBQX_data.groupby(['condition','distance'])['frequency'].describe()
    NBQX_describe['sem'] = NBQX_describe['std'] / np.sqrt(NBQX_describe['count'])
    print(NBQX_describe[['count','mean','sem']])
    sns.barplot(data=NBQX_data, 
                x='distance', 
                y='frequency', 
                errorbar='se', 
                ax=NBQX_plot,
                hue='condition',
                palette=['grey','black'],
                alpha=.5,
                )
    sns.stripplot(data=NBQX_data, 
                x='distance', 
                y='frequency', 
                hue='condition',
                ax=NBQX_plot,
                dodge=True,
                legend=False,
                palette=['grey','black'])
    NBQX_plot.legend(loc='upper right',frameon=False)
    NBQX_plot.set_ylabel('LFP event frequency (Hz)')
    NBQX_plot.set_xlabel(' ')
    # Connect paired observations
    conditions = list(NBQX_data['condition'].unique())
    distances = list(NBQX_data['distance'].unique())
    for i, distance in enumerate(distances):
        subset = NBQX_data[NBQX_data['distance'] == distance]

        for pair_id, pair in subset.groupby('folder'):
            if pair['condition'].nunique() != 2:
                continue

            pair = pair.set_index('condition')

            NBQX_plot.plot(
                [i - 0.2, i + 0.2],
                [pair.loc[conditions[0], 'frequency'],
                pair.loc[conditions[1], 'frequency']],
                color='grey',
                alpha=0.4,
                linewidth=0.8,
                zorder=2
            )
    from scipy.stats import ttest_rel

    for distance, df in NBQX_data.groupby('distance'):

        paired = df.pivot_table(
            index='folder',
            columns='condition',
            values='frequency',
            aggfunc='mean'
        ).dropna(subset=['Pre', 'Post'])

        t, p = ttest_rel(paired['Pre'], paired['Post'])

        print(f"{distance}: n={len(paired)}, t={t:.3f}, p={p:.4f}")


    sns.despine(fig,right=True,top=True)
    fig.savefig(os.path.join(savepath,'Figure_7.svg'))
    plt.show()



datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 7'
Figure7(NBQX_datapath=os.path.join(datapath,'NBQX_results.csv'),
        traces_datapath=os.path.join(datapath,'lfp_windows.csv'),
        cell_datapath=os.path.join(datapath,'results.csv'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 7'
)