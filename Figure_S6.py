
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from code_utils import analysis_plotting as pl

def GenerateFigure(
        datapath,
        savepath,
        ):

    data_to_plot = pd.read_csv(datapath)

    interneuron_subclasses = ['PV','SST','VIP']
    pyramidal_type = data_to_plot['pyramidal_type'].iloc[0] 
    savefig = False  # make true to save individual plots

    fig, ax = plt.subplots(
        3, 3,
        gridspec_kw={
             'width_ratios': [1.2, 0.9, .2],
             'height_ratios': [1, 1, 1],
             'wspace': 0.5,
             'hspace': 0.4
         },
        figsize=(9.27, 10.69)
    )
    labels = ['Cell 1','Cell 2']

        # Remove first column
    for a in ax[:, 0]:
        a.axis('off')

    [PV_scatter, PV_bias,
     SST_scatter, SST_bias,
     VIP_scatter, VIP_bias] = ax[:, 1:].reshape(-1)
    
    # Panel letters on reserved first column + scatter panels
    pl.add_panel_letter(ax[0, 0], 'a')
    pl.add_panel_letter(PV_scatter, 'b', x=-0.3, y=1.15)

    pl.add_panel_letter(ax[1, 0], 'c')
    pl.add_panel_letter(SST_scatter, 'd', x=-0.3, y=1.15)

    pl.add_panel_letter(ax[2, 0], 'e')
    pl.add_panel_letter(VIP_scatter, 'f', x=-0.3, y=1.15)


    interneuron_subclass = 'PV'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    pl.equality_plot(PV_scatter,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(PV_bias,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)

    interneuron_subclass = 'SST'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    pl.equality_plot(SST_scatter,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(SST_bias,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)

    interneuron_subclass = 'VIP'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    pl.equality_plot(VIP_scatter,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(VIP_bias,data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)


    #pl.share_axis([PV_scatter,SST_scatter,VIP_scatter],'y')
    #pl.share_axis([PV_scatter,SST_scatter,VIP_scatter],'x')

    fig.savefig(os.path.join(savepath, 'Figure_S6.svg'), format='svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    GenerateFigure(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 4/Fig_S6_unlabeled_data.csv',
            savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 4',
    )