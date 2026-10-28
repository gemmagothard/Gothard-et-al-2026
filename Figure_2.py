
import matplotlib.pyplot as plt
import os
import numpy as np
from openpyxl import load_workbook
import pandas as pd
import seaborn as sns
#from bioinfokit.analys import stat
# incase any HDF files get stuck open this will close them
import tables
tables.file._open_files.close_all()
from code_utils import analysis_utils as ut
from code_utils import analysis_plotting as pl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from add_example_traces import add_example_to_plot
from pathlib import Path 

import matplotlib as mpl
plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

colors_neurons = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }

colors_pairs = {
                    'OP-OP':'#00CDE5',
                    'IP-OP': '#AE7DF8',
                    'IP-IP':'#E404FF',
                    }


def mannwhitney_posthoc(
    df,
    value_col,
    group_col='pair_type',
    groups=('OP-OP', 'IP-OP', 'IP-IP'),
    correction='holm'
):
    """
    Pairwise Mann-Whitney U tests.

    Returns
    -------
    results : DataFrame
        Test statistics and p-values.

    p_matrix : DataFrame
        Symmetric matrix of corrected p-values.
    """

    comparisons = [
        (groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    ]

    rows = []

    for g1, g2 in comparisons:

        x = df.loc[
            df[group_col] == g1,
            value_col
        ].dropna()

        y = df.loc[
            df[group_col] == g2,
            value_col
        ].dropna()

        U, p = mannwhitneyu(
            x,
            y,
            alternative='two-sided'
        )

        rows.append({
            'group1': g1,
            'group2': g2,
            'U': U,
            'p_raw': p
        })

    results = pd.DataFrame(rows)

    # Correct for multiple comparisons
    results['p_corrected'] = multipletests(
        results['p_raw'],
        method=correction
    )[1]

    # Make symmetric p-value matrix
    p_matrix = pd.DataFrame(
        np.ones((len(groups), len(groups))),
        index=groups,
        columns=groups
    )

    for _, row in results.iterrows():

        p_matrix.loc[
            row['group1'],
            row['group2']
        ] = row['p_corrected']

        p_matrix.loc[
            row['group2'],
            row['group1']
        ] = row['p_corrected']

    return results, p_matrix

if __name__ == "__main__":

    path  = Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 2/')

    ipsc_shared_results = pd.read_csv(path / 'ipsc_results.csv')
    ipsc_results_by_neuron = pd.read_csv(path / 'ipsc_results_by_neuron_2026.csv')

    epsc_shared_results = pd.read_csv(path / 'epsc_results.csv')
    epsc_results_by_neuron = pd.read_csv(path / 'epsc_results_by_neuron_2026.csv')

    ipsc_shared_results = ipsc_shared_results.dropna(subset=['percent_shared_mean'])
    epsc_shared_results = epsc_shared_results.dropna(subset=['percent_shared_mean'])


    # A4 portrait
    fig = plt.figure(figsize=(9.2, 10.69))

    # Main grid
    gs = GridSpec(
        4, 1,
        figure=fig,
        hspace=0.3,
        height_ratios=[1, 1, 1, 1.5]
    )
    # Rows 0–2: 3 columns, first column empty
    gs0 = gs[0].subgridspec(1, 3, wspace=0.2, width_ratios=[0.6, 1, 1]) # row1 
    gs1 = gs[1].subgridspec(1, 3, wspace=0.2, width_ratios=[0.6, 1, 1]) # row2
    gs2 = gs[2].subgridspec(1, 3, wspace=0.2, width_ratios=[0.6, 1, 1]) # row3

    empty1 = fig.add_subplot(gs0[0, 0])
    empty1.axis('off')
    pl.add_panel_letter(fig,gs0[0, 0],'a')

    empty2 = fig.add_subplot(gs1[0, 0])
    empty2.axis('off')
    pl.add_panel_letter(fig,gs1[0, 0],'b')

    empty3 = fig.add_subplot(gs2[0, 0])
    empty3.axis('off')
    pl.add_panel_letter(fig,gs2[0, 0],'c')

    add_example_to_plot(fig,gs0[0, 1],example="OP-OP",type='epsc')
    add_example_to_plot(fig,gs1[0, 1],example="IP-OP",type='epsc')
    add_example_to_plot(fig,gs2[0, 1],example="IP-IP",type='epsc')

    add_example_to_plot(fig,gs0[0, 2],example="OP-OP",type='ipsc')
    add_example_to_plot(fig,gs1[0, 2],example="IP-OP",type='ipsc')
    add_example_to_plot(fig,gs2[0, 2],example="IP-IP",type='ipsc')

    # Row 3: two equal-width plots spanning full figure width
    gs3 = gs[3].subgridspec(1, 2, wspace=0.3)

    epsc_percent_shared = fig.add_subplot(gs3[0, 0])
    ipsc_percent_shared = fig.add_subplot(gs3[0, 1])

    pl.add_panel_letter(fig,gs3[0, 0],'d')
    pl.add_panel_letter(fig,gs3[0, 1],'e')
    
    sns.despine(fig,top=True,right=True)

    #%%%%%%%%%%%% IPSC GRAPHS

    dv = 'percent_shared_mean'
    iv = 'pair_type'
    y_var_label = "Shared inputs (%)"
    sns.barplot(ipsc_shared_results,
                x=iv,
                y=dv,
                hue=iv,
                errorbar='se',
                palette=colors_pairs,
                order=list(colors_pairs.keys()),
                ax=ipsc_percent_shared,
                alpha=.5,
                )
    sns.stripplot(ipsc_shared_results,
                x=iv,
                y=dv,
                hue=iv,
                palette=colors_pairs,
                order=list(colors_pairs.keys()),
                ax=ipsc_percent_shared,
                )
    ipsc_percent_shared.set_ylabel('Shared inhibitory inputs (%)')
    ipsc_percent_shared.set_xlabel(' ')
    categories = ipsc_shared_results[iv].unique()
    test_stat,p, posthoc = pl.perform_stat_test(ipsc_shared_results, dv, iv)
    pl.add_significance_bars(ipsc_percent_shared, posthoc, categories, y_offset=0.05)
    print(test_stat)
    print(p)
    print(posthoc)

    sns.barplot(epsc_shared_results,
                x='pair_type',
                y=dv,
                hue='pair_type',
                errorbar='se',
                palette=colors_pairs,
                order=list(colors_pairs.keys()),
                ax=epsc_percent_shared,
                alpha=.5,
                )
    sns.stripplot(epsc_shared_results,
                x='pair_type',
                y=dv,
                hue='pair_type',
                palette=colors_pairs,
                order=list(colors_pairs.keys()),
                ax=epsc_percent_shared,
                )
    categories = epsc_shared_results[iv].unique()
    test_stat,p, posthoc = pl.perform_stat_test(epsc_shared_results, dv, iv)
    print(test_stat)
    print(p)
    print(posthoc)
    if posthoc is not np.nan:
        pl.add_significance_bars(epsc_percent_shared, posthoc, categories, y_offset=0.1)

    
    epsc_percent_shared.set_ylabel('Shared excitatory inputs (%)')
    epsc_percent_shared.set_xlabel(' ')

    print('IPSC:')
    print(ipsc_shared_results.groupby(['pair_type'])['mouse_id'].nunique())
    print(
        ipsc_shared_results.groupby("pair_type")["percent_shared_mean"]
        .agg(["count", "mean", "std", "sem"])
    )
    
    print('EPSC:')
    print(epsc_shared_results.groupby(['pair_type'])['mouse_id'].nunique())
    print(
        epsc_shared_results.groupby("pair_type")["percent_shared_mean"]
        .agg(["count", "mean", "std", "sem"])
    )
    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 2/Figure_2.svg',dpi=600)
    plt.show()

