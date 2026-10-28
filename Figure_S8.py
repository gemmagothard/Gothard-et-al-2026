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

def FigureS9(
        cell_datapath,
        traces_datapath,
        savepath,
        ):
    

    palette_bars = {'Unlabeled': '#E7E7E7',
                    'IP-derived': '#EAF3E4',
                    'distal': '#FDECCF'}
        
    palette_points = {'Unlabeled': '#808080',
                    'IP-derived': '#8DC170',
                    'distal':'#FACD80'}
    
    order = ['Unlabeled','distal']
    
    data_to_plot = pd.read_csv(cell_datapath)
    data_to_plot = data_to_plot[data_to_plot.cell_type !='IP-derived']

    traces = np.load(traces_datapath)

    fig = plt.figure(figsize=(9.3, 11.69/4))
    gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.3, width_ratios=[2.8,1,0.7])
    it_LFP_examples = fig.add_subplot(gs[0,1])
    it_LFP_averages = fig.add_subplot(gs[0,2],sharey=it_LFP_examples)
    pl.add_panel_letter(fig, gs[0, 0], "a")
    pl.add_panel_letter(fig, gs[0, 1], "b")

    distal_mean = np.mean(traces['distal'],axis=0)
    distal_sem = st.sem(traces['distal'],axis=0)
    unlabeled_mean = np.mean(traces['unlabeled'],axis=0)
    unlabeled_sem = st.sem(traces['unlabeled'],axis=0)

    it_LFP_examples.plot(traces['timevec_trace'],distal_mean,color=palette_points['distal'])
    it_LFP_examples.fill_between(traces['timevec_trace'],distal_mean+distal_sem,distal_mean-distal_sem,color=palette_bars['distal'])
    it_LFP_examples.plot(traces['timevec_trace'],unlabeled_mean,color=palette_points['Unlabeled'])
    it_LFP_examples.fill_between(traces['timevec_trace'],unlabeled_mean+unlabeled_sem,unlabeled_mean-unlabeled_sem,color=palette_bars['Unlabeled'])
    it_LFP_examples.set_ylabel('IT-LFP (-\u03BCV)')
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
    pl.add_significance_bars(this_ax, posthoc, data_to_plot.cell_type.unique(), y_offset=0.05)

    
    this_ax.set_xticks([0,1],['Proximal','Distal'])
    sns.despine(fig,right=True,top=True)
    
    fig.savefig(os.path.join(savepath,'Figure_S8.svg'))
    plt.show()
    


datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 7'
FigureS9(cell_datapath=os.path.join(datapath,'Fig_7_data.csv'),
        traces_datapath=os.path.join(datapath,'Fig_7_IT_LFP_traces.npz'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 7'
)