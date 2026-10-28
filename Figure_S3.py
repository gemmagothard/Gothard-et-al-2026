import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import patsy
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import chi2
from scipy.stats import linregress

plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

from code_utils import analysis_plotting as pl
import figure1_utils as f1_ut

# Get the neuron_type coefficient
def format_and_plot_model_results_(model_results,ax,  y_offset=0.1):
    neuron_term = next(
        name for name in model_results.fe_params.index
        if "neuron_type" in name
    )

    p = model_results.pvalues[neuron_term]

    categories = ["OP-der", "IP-der"]

    posthoc = pd.DataFrame(
        [[1.0, p],
        [p, 1.0]],
        index=categories,
        columns=categories
    )

    pl.add_significance_bars(ax, posthoc, categories, y_offset=y_offset)

    return posthoc

if __name__ == "__main__":

    palette_neurons = {
                    'OP-der':'#00CDE5',
                    'IP-der':'#E404FF',
                    }

    palette = {
                    'OP-OP':'#00CDE5',
                    'IP-OP': '#AE7DF8',
                    'IP-IP':'#E404FF',
                    }
    
    csv_path = Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1')
    
    shared_peaks = pd.read_csv(csv_path / 'shared_peaks_data.csv')
    duration_data = pd.read_csv(csv_path / 'durations_by_state.csv')
    calcium_events = pd.read_csv(csv_path / 'all_calcium_events.csv')

    shared_peaks_cropped = shared_peaks[shared_peaks["XY_distance"].eq(0)]
    print(shared_peaks_cropped.head())

    calcium_events_by_neuron = (
            calcium_events
            .groupby([
                "mouse_id",
                "FOV_id",
                "neuron_id",
                "neuron_uid",
                "neuron_type",
                "peak_state"
            ])
            .agg(
                event_count=("peak_height", "size"),
                mean_peak_height=("peak_height", "median")
            )
            .reset_index())
    
    # Add the appropriate state duration to each row
    calcium_events_by_neuron = calcium_events_by_neuron.merge(
        duration_data,
        left_on=["FOV_id", "peak_state"],
        right_on=["FOV_id", "state_mask"],
        how="left"  
        )
    
    spontaneous_events_by_neuron = calcium_events_by_neuron[calcium_events_by_neuron.peak_state=='spontaneous']
    spontaneous_events_by_neuron["Mean event amplitude (ΔF/F)"] = pd.to_numeric(
                    spontaneous_events_by_neuron["mean_peak_height"],
                    errors="coerce"
                )
    spontaneous_events_by_neuron['Event rate (min$^{-1}$)'] = spontaneous_events_by_neuron.event_count / (spontaneous_events_by_neuron.state_duration / 60)
    print(spontaneous_events_by_neuron.groupby('neuron_type').size())

    fig = plt.figure(figsize=(10, 11.69*0.6))

    # Main grid: 2 rows
    gs = fig.add_gridspec(
        2, 1,
        hspace=0.5
    )
    # Top row: 1 × 3
    gs_top = gs[0].subgridspec(1, 3, wspace=0.5)

    dist = fig.add_subplot(gs_top[0, 0])
    event_rate = fig.add_subplot(gs_top[0, 1])
    event_amp = fig.add_subplot(gs_top[0, 2])

    # Bottom row: 1 × 2
    gs_bottom = gs[1].subgridspec(1, 2, wspace=0.4)

    amp_scatter = fig.add_subplot(gs_bottom[0, 0])
    rate_scatter = fig.add_subplot(gs_bottom[0, 1])

    pl.add_panel_letter(fig,gs_top[0, 0],'a')
    pl.add_panel_letter(fig,gs_top[0, 1],'b')
    pl.add_panel_letter(fig,gs_top[0, 2],'c')
    pl.add_panel_letter(fig,gs_bottom[0, 0],'d')
    pl.add_panel_letter(fig,gs_bottom[0, 1],'e')
    
    y = 'Mean event amplitude (ΔF/F)'
    sns.barplot(spontaneous_events_by_neuron,
                hue='neuron_type',
                y=y,
                x='neuron_type',
                ax=event_amp,
                errorbar='se',
                palette=palette_neurons
                )
    event_amp.set_xlabel(' ')
    print(y)
    test_stat,p, posthoc = pl.perform_stat_test(spontaneous_events_by_neuron, y, 'neuron_type')
    print(test_stat)
    print(posthoc)
    pl.add_significance_bars(event_amp, posthoc, spontaneous_events_by_neuron.neuron_type.unique(), y_offset=0.05)

    y = 'Event rate (min$^{-1}$)'
    sns.barplot(spontaneous_events_by_neuron,
                hue='neuron_type',
                y=y,
                x='neuron_type',
                palette=palette_neurons,
                errorbar='se',
                ax=event_rate)
    event_rate.set_xlabel(' ')
    print(y)
    test_stat,p, posthoc = pl.perform_stat_test(spontaneous_events_by_neuron, y, 'neuron_type')
    print(test_stat)
    print(posthoc)
    pl.add_significance_bars(event_rate, posthoc, spontaneous_events_by_neuron.neuron_type.unique(), y_offset=0.05)
    
    y = 'XY_dist_raw'
    sns.barplot(shared_peaks_cropped,
                hue='lineage_label',
                x='lineage_label',
                y=y,
                palette=palette,
                order = list(palette.keys()),
                errorbar='se',
                ax=dist)
    dist.set_ylabel('Distance between pairs (µm)')
    dist.set_xlabel(' ')
    print(y)
    test_stat,p, posthoc = pl.perform_stat_test(shared_peaks_cropped, y, 'lineage_label')
    print(test_stat)
    print(posthoc)
    #pl.add_significance_bars(event_rate, posthoc, shared_peaks_cropped.lineage_label.unique(), y_offset=0.05)



    # Fixed effects
    y, X = patsy.dmatrices(
        """
        XY_dist_raw ~
        C(lineage_label, Treatment(reference='OP-OP'))
        + C(mouse_id)
        """,
        data=shared_peaks,
        return_type="dataframe"
    )

    # FOV random intercept
    groups = shared_peaks["FOV_id"]

    # Multi-membership neuron random effect
    # Accounts for neuron_a / neuron_b appearing in multiple pairs
    vc = f1_ut.make_neuron_vc(shared_peaks)
    model_dist = sm.MixedLM(
        endog=y,
        exog=X,
        groups=groups,
        exog_re=np.ones((len(shared_peaks), 1)),
        exog_vc=vc
    )
    result_dist = model_dist.fit(reml=True)
    print(result_dist.summary())
    print("\nExact p-values:")
    print(result_dist.pvalues[result_dist.fe_params.index])




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
    lineage_has_effect = test_lineage_effect(result_dist)

    results, posthoc = f1_ut.pairwise_lineage_tests(result_dist)
    print(results)
    categories = ["OP-OP", "IP-IP", "IP-OP"]
    pl.add_significance_bars(dist, posthoc, categories, y_offset=1)

    y = shared_peaks_cropped['events_above_chance']
    event_label = 'Above chance co-activity rate (min$^{-1}$)'

    # Mean peak amplitude
    x = shared_peaks_cropped['mean_peak_amp']

    slope, intercept, r, p, se = linregress(x, y)

    sns.regplot(data=shared_peaks_cropped,
                x='mean_peak_amp',
                y='events_above_chance',
                scatter_kws={'alpha': 0.05},
                ax=amp_scatter
                )
    amp_scatter.set_xlabel("Mean event amplitude (ΔF/F)")
    amp_scatter.set_ylabel(event_label)
    amp_scatter.set_title(f'R² = {r**2:.3f} \np = {p:.3g}',fontsize=10)

    # Mean peak frequency
    x = shared_peaks_cropped['mean_event_freq']
    slope, intercept, r, p, se = linregress(x, y)
    sns.regplot(
        data=shared_peaks_cropped,
        x='mean_event_freq',
        y='events_above_chance',
        scatter_kws={'alpha': 0.05},
        ax=rate_scatter
    )
    rate_scatter.set_xlabel('Event rate (min$^{-1}$)')
    rate_scatter.set_ylabel(event_label)
    rate_scatter.set_title(f'R² = {r**2:.3f} \np = {p:.3g}',fontsize=10)

    sns.despine(fig,right=True,top=True)
    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1/figure_S3.svg',format='svg',bbox_inches=None)
    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1/figure_S3.png')
    fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 1/figure_S3.tiff',dpi=600)
    plt.show()