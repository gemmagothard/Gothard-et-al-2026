
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
import itertools as it 
import patsy
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
import scipy.stats as st


import figure1_utils as f1_ut
from code_utils import analysis_plotting as pl

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

def add_scale_bar(
    ax,
    x_size=10,
    y_size=1,
    x_label=None,
    y_label=None,
    x=1.02,
    y=0.02,
    lw=2,
    fontsize=8,
):
    """
    Add an L-shaped scale bar just outside a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add the scale bar to.
    x_size : float
        Horizontal scale bar length in x-axis data units.
    y_size : float
        Vertical scale bar length in y-axis data units.
    x_label : str, optional
        Label for horizontal bar. Defaults to str(x_size).
    y_label : str, optional
        Label for vertical bar. Defaults to str(y_size).
    location : {"lower right", "lower left"}
        Scale bar location.
    pad : float
        Fraction of axis range used as padding from the edges.
    lw : float
        Scale bar line width.
    fontsize : float
        Label font size.
    x_size, y_size
        Scale-bar lengths in data units.
    x, y
        Position of the lower-left corner in axes coordinates.
        x > 1 places it to the right of the plot.
    """


    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert data-unit lengths to axes-coordinate lengths
    x_frac = x_size / (xlim[1] - xlim[0])
    y_frac = y_size / (ylim[1] - ylim[0])

    if x_label is None:
        x_label = str(x_size)

    if y_label is None:
        y_label = str(y_size)

    # horizontal bar
    ax.plot(
        [x, x + x_frac],
        [y, y],
        transform=ax.transAxes,
        color="black",
        lw=lw,
        solid_capstyle="butt",
        clip_on=False,
    )

    # vertical bar
    ax.plot(
        [x, x],
        [y, y + y_frac],
        transform=ax.transAxes,
        color="black",
        lw=lw,
        solid_capstyle="butt",
        clip_on=False,
    )

    # horizontal label
    ax.text(
        x + x_frac / 2,
        y - 0.02,
        x_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        clip_on=False,
    )

    # vertical label
    ax.text(
        x - 0.015,
        y + y_frac / 2,
        y_label,
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize,
        clip_on=False,
    )

if __name__ == "__main__":

    palette_bars = {
                    'OP-OP':'#00CDE5',
                    'IP-OP': '#AE7DF8',
                    'IP-IP':'#E404FF',
                    }
    
    palette_neurons = {
                    'OP-der':'#00CDE5',
                    'IP-der':'#E404FF',}
    
    csv_path = Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1')
    
    shared_peaks = pd.read_csv(csv_path / 'shared_peaks_data.csv')
    pial_dist = pd.read_csv(csv_path / 'pial_dist.csv')
    pial_dist_by_brain = pial_dist.groupby(['folder','cell_type'])['pial_dist'].mean().reset_index()
    print(pial_dist.groupby(['cell_type']).count())
    pial_dist_describe = pial_dist_by_brain.groupby(['cell_type'])['pial_dist'].describe()
    print(pial_dist_describe)
    pial_dist_describe['sem'] = pial_dist_describe['std'] / np.sqrt(pial_dist_describe['count'])
    print(pial_dist_describe)

    shared_peaks_cropped = shared_peaks[shared_peaks["XY_distance"].eq(0)]

    describe = shared_peaks_cropped.groupby(['lineage_label'])['events_above_chance'].describe()
    describe['sem'] = describe['std'] / np.sqrt(describe['count'])
    print(describe[['count','mean','std','sem']])

    def print_example_candidates(shared_peaks):
        for_examples = shared_peaks[
        shared_peaks['dff_peaks_a'].between(4, 6) &
        shared_peaks['dff_peaks_b'].between(4, 6) &
        shared_peaks['pairwise_frequency_difference'].between(0,10)
        ]

        highest = for_examples.sort_values(
        ["events_above_chance", "pairwise_frequency_difference", "duration" ],
        ascending=[False, True, False]
        )
        print('OP-OP pairs')
        op_op = highest[highest.lineage_label == 'OP-OP']

        start = int(len(op_op) * 0.15)
        end = int(len(op_op) * 0.3)

        print(op_op.iloc[start:end])

        lowest = for_examples.sort_values(
        ["events_above_chance","pairwise_frequency_difference","duration"],
        ascending=[True, True,False]
        )
        print('IP-OP pairs')
        print(lowest[lowest.lineage_label=='IP-OP'].head(20))
        print('IP-IP pairs')
        print(lowest[lowest.lineage_label=='IP-IP'].head(20))

    
    fig = plt.figure(figsize=(9, 11.69))  # wxh, increase by 1inch from affinity designer measurements to account for white space border

    gs = GridSpec(
        4, 1,
        figure=fig,
        hspace=0.3,
        height_ratios=[1.4,1.3,1.2,1.4],
    )

    # first row 
    gs0 = gs[0].subgridspec(1, 3,width_ratios=[4,1,0.5])
    pl.add_panel_letter(fig,gs0[0], 'a')
    pl.add_panel_letter(fig,gs0[1], 'b')
    pial_dist_bar = fig.add_subplot(gs0[2])

    # second row
    gs1 = gs[1].subgridspec(1, 2,width_ratios=[1,5])
    pl.add_panel_letter(fig,gs1[0], 'c')
    pl.add_panel_letter(fig,gs1[1], 'd')

    # third row 
    gs2 = gs[2].subgridspec(1,3)
    gs_op_op = gs2[0].subgridspec(2, 1, hspace=0.2, wspace=0)
    gs_op_ip = gs2[1].subgridspec(2, 1, hspace=0.2, wspace=0)
    gs_ip_ip = gs2[2].subgridspec(2, 1, hspace=0.2, wspace=0)

    # fourth row
    gs3 = gs[3].subgridspec(1,3,wspace=0.4)

    event_label = 'Above chance co-activity rate (min$^{-1}$)'

    # OP-OP
    op_op_top = fig.add_subplot(gs_op_op[0, 0])
    op_op_bottom = fig.add_subplot(gs_op_op[1, 0])
    # IP-OP
    op_ip_top = fig.add_subplot(gs_op_ip[0, 0])
    op_ip_bottom = fig.add_subplot(gs_op_ip[1, 0])
    # IP-IP
    ip_ip_top = fig.add_subplot(gs_ip_ip[0, 0])
    ip_ip_bottom = fig.add_subplot(gs_ip_ip[1, 0])

    f1_ut.add_example_trace(op_op_top, op_op_bottom, 'OP-OP')
    f1_ut.add_example_trace(op_ip_top, op_ip_bottom, 'IP-OP')
    f1_ut.add_example_trace(ip_ip_top, ip_ip_bottom, 'IP-IP')

    example_axis_list = [op_op_top,op_op_bottom,
                   op_ip_top,op_ip_bottom,
                   ip_ip_top,ip_ip_bottom]

    pl.share_axis(example_axis_list,'y')
    # add figure-level legend
    f1_ut.compile_legends(example_axis_list,fig,y_pos=0.32)

    # row 1 pial dist
    sns.violinplot(pial_dist,
                   x='cell_type',
                   y='pial_dist',
                   ax=pial_dist_bar,
                   hue='cell_type',
                   split=True,
                   inner='box',
                   palette=palette_neurons)
    OP_pial_dist = pial_dist_by_brain[pial_dist_by_brain.cell_type=='OP-der'].pial_dist
    IP_pial_dist = pial_dist_by_brain[pial_dist_by_brain.cell_type=='IP-der'].pial_dist
    # labels = [tick.get_text() for tick in pial_dist_bar.get_xticklabels()]
    # positions = pial_dist_bar.get_xticks()
    # x_pos = dict(zip(labels, positions))
    # pial_dist_bar.errorbar(
    #     [x_pos['OP-der'], x_pos['IP-der']],
    #     [OP_pial_dist.mean(), IP_pial_dist.mean()],
    #     yerr=[st.sem(OP_pial_dist), st.sem(IP_pial_dist)],
    #     fmt='o',
    #     color='black',
    #     markersize=2,
    #     capsize=0,
    #     zorder=10
    # )
    pial_dist_bar.xaxis.set_visible(False)
    sns.despine(fig,pial_dist_bar,bottom=True)
    pial_dist_bar.invert_yaxis()
    pial_dist_bar.set_ylabel('Pial distance (\u03BCm)')
    t_stat, p_value = st.ttest_rel(OP_pial_dist,
                                   IP_pial_dist)
    print(f"Pial dist t-test: t = {t_stat:.3f}, p = {p_value:.3e}")
    # Symmetrical p-value matrix
    posthoc = pd.DataFrame(
        [[1, p_value],
        [p_value, 1]],
        index=['IP-der', 'OP-der'],
        columns=['IP-der', 'OP-der']
    )
    pl.add_significance_bars(pial_dist_bar, posthoc, pial_dist.cell_type.unique(), y_offset=0.6)

    # Row 3
    distance_plot = fig.add_subplot(gs3[0])
    cumulative_plot = fig.add_subplot(gs3[1])
    bar_plot = fig.add_subplot(gs3[2])

    pl.add_panel_letter(fig,gs_op_op[0, 0], 'e')
    pl.add_panel_letter(fig,gs3[0], "f")
    pl.add_panel_letter(fig,gs3[1], "g")
    pl.add_panel_letter(fig,gs3[2], "h")

    sns.ecdfplot(shared_peaks_cropped,
            hue='lineage_label',
            palette=palette_bars,
            stat='proportion',
            ax=cumulative_plot,
            hue_order = list(palette_bars.keys()),
            x='events_above_chance')
    #cumulative_plot.set_ylim(0.05,0.95)
    cumulative_plot.set_xlim(-1,4)

    cumulative_plot.set_xlabel(event_label)
    sns.move_legend(
        cumulative_plot,
        "center right",
        title=None,
        frameon=False
    )

    sns.barplot(
        data=shared_peaks_cropped,
        x="lineage_label",
        y="events_above_chance",
        hue='lineage_label',
        palette=palette_bars,
        ax=bar_plot,
        order = list(palette_bars.keys()),
        legend=False,
        errorbar='se',
    )
    bar_plot.set_ylabel(event_label)
    bar_plot.set_xlabel(' ')

    sns.lineplot(shared_peaks,
                 hue='lineage_label',
                 palette=palette_bars,
                 x='XY_distance',
                 y='events_above_chance',
                 ax=distance_plot,
                 errorbar='se',
                 )
    distance_plot.legend(title=None,frameon=False)
    distance_plot.set_ylabel(event_label)
    distance_plot.set_xlabel('Distance between pairs (µm)')
    
    # model with no interaction to test if distance affects lineage
    # Fixed effects - distance only (no interaction)
    y, X = patsy.dmatrices(
        """
        events_above_chance ~ XY_distance
        + C(lineage_label, Treatment(reference='OP-OP'))
        + C(mouse_id)
        """,
        data=shared_peaks,
        return_type="dataframe"
    )
    # FOV random intercept
    groups = shared_peaks["FOV_id"]
    vc = f1_ut.make_neuron_vc(shared_peaks)

    lineage_distance_model = sm.MixedLM(
        endog=y,
        exog=X,
        groups=groups,
        exog_re=np.ones((len(shared_peaks), 1)),
        exog_vc=vc
    )

    lineage_distance_results = lineage_distance_model.fit(reml=True)
    print('results from no interaction model')
    print(f"β = {lineage_distance_results.params['XY_distance']:.4f}")
    print(f"z = {lineage_distance_results.tvalues['XY_distance']:.2f}")
    print(f"P = {lineage_distance_results.pvalues['XY_distance']:.2e}")

    def perform_lm_nested(df,ax):

        # Fixed effects
        y, X = patsy.dmatrices(
            """events_above_chance ~ 
            C(lineage_label, Treatment(reference='OP-OP'))
            +C(mouse_id)""",
            data=df,
            return_type="dataframe"
        )

        # FOV random intercept
        groups = df["FOV_id"]
        vc = f1_ut.make_neuron_vc(df)
        lineage_model = sm.MixedLM(
            endog=y,
            exog=X,
            groups=groups,
            exog_re=np.ones((len(df), 1)),
            exog_vc=vc
        )
        lineage_model_results = lineage_model.fit(reml=True)
        print(f"β = {lineage_model_results.params}")
        print(f"z = {lineage_model_results.tvalues}")

        # initial test to see if there are differences - Wald chi squared
        def test_lineage_effect(result):

            beta = result.fe_params
            cov = result.cov_params().loc[
                beta.index,
                beta.index
            ]

            names = list(beta.index)
            n = len(names)

            # Find lineage coefficient positions
            ipip_idx = next(
                i for i, name in enumerate(names)
                if "IP-IP" in name
            )

            ipop_idx = next(
                i for i, name in enumerate(names)
                if "IP-OP" in name
            )

            # --------------------------------------------------
            # OMNIBUS TEST
            #
            # H0:
            #   beta_IP-IP = 0
            #   beta_IP-OP = 0
            #
            # Equivalent to:
            #   OP-OP = IP-IP = IP-OP
            # --------------------------------------------------

            R = np.zeros((2, n))
            R[0, ipip_idx] = 1
            R[1, ipop_idx] = 1

            beta_r = beta.values
            cov_r = cov.values

            Rb = R @ beta_r
            RVR = R @ cov_r @ R.T

            chi2_stat = float(
                Rb.T @ np.linalg.inv(RVR) @ Rb
            )

            df = 2

            p_value = chi2.sf(
                chi2_stat,
                df
            )

            print("\n====================================")
            print("OMNIBUS LINEAGE TEST")
            print("====================================")
            print(f"Chi-square = {chi2_stat:.3f}")
            print(f"df         = {df}")
            print(f"p-value    = {p_value:.4g}")

            # --------------------------------------------------
            # PAIRWISE COMPARISONS
            # --------------------------------------------------

        test_lineage_effect(lineage_model_results)
        results, posthoc = f1_ut.pairwise_lineage_tests(lineage_model_results)
        categories = ["OP-OP", "IP-IP", "IP-OP"]
        print(posthoc)
        pl.add_significance_bars(ax, posthoc, categories, y_offset=0.1)

    perform_lm_nested(shared_peaks_cropped,bar_plot)

    sns.despine(fig,top=True,right=True)

    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1/figure_1.svg',format='svg',bbox_inches=None)
    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1/figure_1.png')

    plt.show()


