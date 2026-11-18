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

def GenerateFigure(
        cell_datapath,
        traces_datapath,
        savepath,
        ):

    palette_bars = {'Unlabeled': '#E7E7E7',
                    'IP-der': '#EAF3E4',
                    'IP-derived': '#EAF3E4',
                    'distal': '#FDECCF'}
        
    palette_points = {'Unlabeled': '#808080',
                    'IP-der': '#8DC170',
                    'IP-derived': '#8DC170',
                    'distal':'#FACD80'}
    
    order = ['Unlabeled','distal']
    order_ip_der = ['Unlabeled','IP-der']
    
    data_to_plot = pd.read_csv(cell_datapath)
    data_ip_der = data_to_plot[data_to_plot.cell_type !='distal']
    data_to_plot = data_to_plot[data_to_plot.cell_type !='IP-der']
    print(data_to_plot)

    traces = pd.read_csv(traces_datapath)
    traces = traces[traces.cell_type.isin(['distal','Unlabeled'])]

    fig = plt.figure(figsize=(9.3, 11.69/2))

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=0.5
    )

    # Top row: 1x3 with original width ratios
    gs_top = gs[0].subgridspec(
        1, 3,
        width_ratios=[2.8, 1, 0.7],
        wspace=0.2
    )
    # Bottom row: 1x4
    gs_bottom = gs[1].subgridspec(
        1, 4,
        wspace=0.5
    )

    it_LFP_examples = fig.add_subplot(gs_top[1])
    it_LFP_averages = fig.add_subplot(gs_top[2],sharey=it_LFP_examples)

    input_amp_ax = fig.add_subplot(gs_bottom[0])
    input_freq_ax = fig.add_subplot(gs_bottom[1])
    lfp_amp_ax = fig.add_subplot(gs_bottom[2])
    lfp_freq_ax = fig.add_subplot(gs_bottom[3])

    pl.add_panel_letter(fig, gs_top[0, 0], "a")
    pl.add_panel_letter(fig, gs_top[0, 1], "b")
    pl.add_panel_letter(fig, gs_top[0, 1], "b")

    pl.add_panel_letter(fig, gs_bottom[0, 0], "c")
    pl.add_panel_letter(fig, gs_bottom[0, 1], "d")
    pl.add_panel_letter(fig, gs_bottom[0, 2], "e")
    pl.add_panel_letter(fig, gs_bottom[0, 3], "f")

    sns.lineplot(traces,
             x='timevec',
             y='lfp_windows',
             hue='cell_type',
             palette=palette_points,
             errorbar='se',
             estimator='mean',
             ax=it_LFP_examples,
             legend=False,
             alpha=1
             )
    it_LFP_examples.set_ylabel('IT-LFP (-\u03BCV)')
    it_LFP_examples.set_xlabel('Time lags (ms)')
    it_LFP_examples.axvline(0,linestyle='dotted',color='black')
    it_LFP_examples.set_xlim(-50,50)

    describe = data_to_plot.groupby(['cell_type'])['IT_lfp'].describe()
    describe['sem'] = describe['std'] / np.sqrt(describe['count'])

    print(describe)


    y = 'IT_lfp'
    this_ax = it_LFP_averages
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel(' ')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                errorbar='se',
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax,
                alpha=.7)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'cell_type')
    print(test_stat)
    print(posthoc)
    print(f'{y}, T = {test_stat:3g}, p = {p:3g}')
    pl.add_significance_bars(this_ax, posthoc, data_to_plot.cell_type.unique(), y_offset=0.05)
    this_ax.set_xticks([0,1],['Proximal','Distal'])

    y = 'input_amp'
    this_ax = input_amp_ax
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Synaptic input amplitude (nS)')
    sns.barplot(data=data_ip_der,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                errorbar='se',
                order = order_ip_der,
                edgecolor='black',)
    sns.stripplot(data=data_ip_der,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order_ip_der,
                ax=this_ax,
                alpha=.7)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'cell_type')
    print(f'{y}, T = {test_stat:3g}, p = {p:3g}')

    y = 'input_freq'
    this_ax = input_freq_ax
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Synaptic input frequency (Hz)')
    sns.barplot(data=data_ip_der,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                errorbar='se',
                order = order_ip_der,
                edgecolor='black',)
    sns.stripplot(data=data_ip_der,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order_ip_der,
                ax=this_ax,
                alpha=.7)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'cell_type')
    print(f'{y}, T = {test_stat:3g}, p = {p:3g}')


    y = 'lfp_amp'
    this_ax = lfp_amp_ax
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('LFP event amplitude (-\u03BCV)')
    sns.barplot(data=data_ip_der,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                errorbar='se',
                order = order_ip_der,
                edgecolor='black',)
    sns.stripplot(data=data_ip_der,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order_ip_der,
                ax=this_ax,
                alpha=.7)
    test_stat,p, posthoc = pl.perform_stat_test(data_ip_der, y, 'cell_type')
    print(f'{y}, T = {test_stat:3g}, p = {p:3g}')


    y = 'lfp_freq'
    this_ax = lfp_freq_ax
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('LFP event frequency (Hz)')
    sns.barplot(data=data_ip_der,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                errorbar='se',
                order = order_ip_der,
                edgecolor='black',)
    sns.stripplot(data=data_ip_der,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order_ip_der,
                ax=this_ax,
                alpha=.7)
    test_stat,p, posthoc = pl.perform_stat_test(data_to_plot, y, 'cell_type')
    print(f'{y}, T = {test_stat:3g}, p = {p:3g}')




    sns.despine(fig,right=True,top=True)
    
    fig.savefig(os.path.join(savepath,'Figure_S8.svg'))
    plt.show()
    


datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 7'
GenerateFigure(cell_datapath=os.path.join(datapath,'results.csv'),
        traces_datapath=os.path.join(datapath,'lfp_windows.csv'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 7'
)