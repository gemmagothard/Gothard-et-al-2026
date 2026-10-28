
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf
from code_utils import analysis_plotting as pl

def Figure4(
        datapath,
        savepath,
        ):

    data_to_plot = pd.read_csv(datapath)
    data_to_plot["bias_centered"] = data_to_plot["bias_index"] - 0.5

    pyramidal_type = data_to_plot['pyramidal_type'].iloc[0] 
    savefig = False  # make true to save individual plots
    palette = ['#00CDE5', '#E404FF'] 

    fig = plt.figure(figsize=(9.8, 11.69))

    gs = fig.add_gridspec(
        4, 4,
        width_ratios=[0.6, 1, 0.2, 1],
        height_ratios=[1, 1, 1, 1],
        wspace=0.5,
        hspace=0.4
    )

    # Panel letter for first row
    empty_row = fig.add_subplot(gs[0, 0])
    empty_row.axis('off')
    pl.add_panel_letter(fig,gs[0, 0], 'a')

    PV_scatter = fig.add_subplot(gs[1,1])
    PV_bias = fig.add_subplot(gs[1,2])
    PV_dist = fig.add_subplot(gs[1,3])

    SST_scatter = fig.add_subplot(gs[2,1])
    SST_bias = fig.add_subplot(gs[2,2])
    SST_dist = fig.add_subplot(gs[2,3])

    VIP_scatter = fig.add_subplot(gs[3,1])
    VIP_bias = fig.add_subplot(gs[3,2])
    VIP_dist = fig.add_subplot(gs[3,3])

    # PV row letters
    pl.add_panel_letter(fig,gs[1,0],'b')
    pl.add_panel_letter(fig,gs[1,1], 'c')
    pl.add_panel_letter(fig,gs[1,3], 'd')

    # SST row letters
    pl.add_panel_letter(fig,gs[2,0],'e')
    pl.add_panel_letter(fig,gs[2,1], 'f')
    pl.add_panel_letter(fig,gs[2,3], 'g')

    # VIP row letters
    pl.add_panel_letter(fig,gs[3,0],'h')
    pl.add_panel_letter(fig,gs[3,1], 'i')
    pl.add_panel_letter(fig,gs[3,3], 'j')


    labels = ['OP-der','IP-der']

    interneuron_subclass = 'PV'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    model = smf.mixedlm(
        "bias_centered ~ 1",
        data=data_subset,
        groups=data_subset["mouse_id"]
    )
    result = model.fit()
    pl.equality_plot(PV_scatter,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(PV_bias,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)
    pl.scatter_bar_plot(PV_dist,[data_subset.OP_pial_dist.values,data_subset.IP_pial_dist.values], palette, labels, True ,"Pial distance (\u03BCm)", savefig, savepath,False)

    interneuron_subclass = 'SST'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    model = smf.mixedlm(
        "bias_centered ~ 1",
        data=data_subset,
        groups=data_subset["mouse_id"]
    )
    result = model.fit()
    pl.equality_plot(SST_scatter,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(SST_bias,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)
    pl.scatter_bar_plot(SST_dist,[data_subset.OP_pial_dist.values,data_subset.IP_pial_dist.values], palette, labels, True ,"Pial distance (\u03BCm)", savefig, savepath,False)

    interneuron_subclass = 'VIP'
    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    model = smf.mixedlm(
        "bias_centered ~ 1",
        data=data_subset,
        groups=data_subset["mouse_id"]
    )
    result = model.fit()
    pl.equality_plot(VIP_scatter,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
    pl.biasplot(VIP_bias,data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)
    pl.scatter_bar_plot(VIP_dist,[data_subset.OP_pial_dist.values,data_subset.IP_pial_dist.values], palette, labels, True ,"Pial distance (\u03BCm)", savefig, savepath,False)


    pl.share_axis([PV_scatter,SST_scatter,VIP_scatter],'y')
    pl.share_axis([PV_scatter,SST_scatter,VIP_scatter],'x')

    fig.savefig(os.path.join(savepath, 'Figure_4.svg'), format='svg', bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    Figure4(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 4/Fig_4_RG_data.csv',
            savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 4',
    )
