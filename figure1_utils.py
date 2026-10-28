import pandas as pd
import numpy as np
from pathlib import Path
import itertools as it
from statsmodels.regression.mixed_linear_model import VCSpec
from statsmodels.stats.multitest import multipletests

colors = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }


import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from code_utils import analysis_plotting as pl
import statsmodels.formula.api as smf


def pairwise_to_matrix(posthoc, categories, p_col="p_holm"):
        """
        Convert pairwise comparison results into a symmetric
        p-value matrix for add_significance_bars().
        """
        matrix = pd.DataFrame(
            np.ones((len(categories), len(categories))),
            index=categories,
            columns=categories
        )

        for _, row in posthoc.iterrows():
            cat1, cat2 = row["comparison"].split(" vs ")
            p = row[p_col]

            matrix.loc[cat1, cat2] = p
            matrix.loc[cat2, cat1] = p

        return matrix

def mixedlm_pairwise_pvalues(result, categories, correction='holm'):
    
    if len(categories)>2:

        names = list(result.fe_params.index)

        # Find coefficient names
        ipip_name = next(
            x for x in names if '[T.IP-IP]' in x
        )
        ipop_name = next(
            x for x in names if '[T.IP-OP]' in x
        )

        # OP-OP comparisons directly from model
        p_ipip_opop = result.pvalues[ipip_name]
        p_ipop_opop = result.pvalues[ipop_name]

        # IP-IP vs IP-OP contrast
        L = np.zeros((1, len(result.fe_params)))

        L[0, names.index(ipip_name)] = 1
        L[0, names.index(ipop_name)] = -1

        contrast = result.t_test(L)

        p_ipip_ipop = float(
            np.asarray(contrast.pvalue).squeeze()
        )

        # Collect results
        posthoc = pd.DataFrame({
            'comparison': [
                'IP-IP vs OP-OP',
                'IP-OP vs OP-OP',
                'IP-IP vs IP-OP'
            ],
            'p_raw': [
                p_ipip_opop,
                p_ipop_opop,
                p_ipip_ipop
            ]
        })

        # Multiple-comparison correction
        posthoc['p_corrected'] = multipletests(
            posthoc['p_raw'],
            method=correction
        )[1]

    if len(categories) == 2:

        names = result.fe_params.index.tolist()

        ip_name = next(
            x for x in names
            if '[T.IP-der]' in x
        )

        p_ip_vs_op = result.pvalues[ip_name]

        posthoc = pd.DataFrame({
            "comparison": ["IP-der vs OP-der"],
            "p_raw": [p_ip_vs_op],
            "p_corrected": [p_ip_vs_op]
        })


    return posthoc

def perform_mixed_lm_stats(df, dv, iv, group_var, ax):
    """
    Perform mixed linear model analysis and print results.

    Parameters
    ----------
    df : DataFrame
        Input data.
    dv : str
        Dependent variable.
    iv : str
        Independent / Fixed effect variable.
    group_var : str
        Grouping variable for random effects.

    Returns
    -------
    None
    """

    df = df.dropna(subset=[dv, group_var, iv])

    if (iv == 'pair_type') or (iv == 'lineage_label'):
        categories = ["OP-OP", "IP-IP", "IP-OP"]
        reference_group = 'OP-OP'
    elif iv == 'neuron_type':
        categories = ["OP-der", "IP-der"]
        reference_group = 'OP-der'
    elif iv == 'condition':
        categories = ["OP-der", "IP-der"]
        reference_group = 'OP-der'

    model = smf.mixedlm(
    f'{dv} ~ C({iv}, Treatment(reference="{reference_group}"))',
    data=df,
    groups=df[group_var]
    )
    result = model.fit()

    print(result.summary())

    posthoc = mixedlm_pairwise_pvalues(result,categories,correction='holm')
    posthoc_matrix = pairwise_to_matrix(
        posthoc,
        categories=categories,
        p_col="p_corrected"
    )

    pl.add_significance_bars(ax, posthoc_matrix, categories, y_offset=0.05)

def pairwise_lineage_tests(result):

    beta = result.fe_params
    names = list(beta.index)
    n = len(names)

    # Find lineage coefficients
    ipip_idx = next(
        i for i, name in enumerate(names)
        if "lineage_label" in name and "IP-IP" in name
    )

    ipop_idx = next(
        i for i, name in enumerate(names)
        if "lineage_label" in name and "IP-OP" in name
    )

    contrasts = {
        "OP-OP vs IP-IP": {
            ipip_idx: -1
        },
        "OP-OP vs IP-OP": {
            ipop_idx: -1
        },
        "IP-IP vs IP-OP": {
            ipip_idx: 1,
            ipop_idx: -1
        }
    }

    results = []

    for comparison, weights in contrasts.items():

        L = np.zeros(n)

        for idx, weight in weights.items():
            L[idx] = weight

        test = result.t_test(L[np.newaxis, :])

        estimate = float(np.asarray(test.effect).squeeze())
        se = float(np.asarray(test.sd).squeeze())
        p = float(np.asarray(test.pvalue).squeeze())

        results.append({
            "comparison": comparison,
            "difference": estimate,
            "SE": se,
            "CI_low": estimate - 1.96 * se,
            "CI_high": estimate + 1.96 * se,
            "p": p
        })

    pairwise = pd.DataFrame(results)

    # Holm multiple-comparison correction
    pairwise["p_holm"] = multipletests(
        pairwise["p"],
        method="holm"
    )[1]

    # ----------------------------------------
    # Create posthoc matrix for plotting
    # ----------------------------------------

    categories = ["OP-OP", "IP-IP", "IP-OP"]

    posthoc = pd.DataFrame(
        np.ones((len(categories), len(categories))),
        index=categories,
        columns=categories
    )

    for _, row in pairwise.iterrows():

        cat1, cat2 = row["comparison"].split(" vs ")
        p = row["p_holm"]

        posthoc.loc[cat1, cat2] = p
        posthoc.loc[cat2, cat1] = p

    return pairwise, posthoc

def make_neuron_vc(df, group_col="FOV_id"):
        """
        Create a multi-membership random effect for neuron pairs.

        Each pair gets:
            +1 * u_neuron_a
            +1 * u_neuron_b

        so the same neuron has the same random effect regardless
        of whether it appears as neuron_a or neuron_b.
        """
        mats = []
        colnames = []

        # VCSpec requires groups/matrices in sorted order
        for fov in sorted(df[group_col].unique()):

            d = df[df[group_col] == fov]

            neurons = sorted(
                set(d["neuron_a_uid"]) |
                set(d["neuron_b_uid"])
            )

            neuron_to_col = {
                neuron: i
                for i, neuron in enumerate(neurons)
            }

            mat = np.zeros((len(d), len(neurons)))

            for row_idx, (_, row) in enumerate(d.iterrows()):
                mat[row_idx, neuron_to_col[row["neuron_a_uid"]]] = 1
                mat[row_idx, neuron_to_col[row["neuron_b_uid"]]] = 1

            mats.append(mat)
            colnames.append(neurons)

        return VCSpec(
            names=["neuron"],
            colnames=[colnames],
            mats=[mats]
        )

def scale_bar(ax, x_scale=5, y_scale=2, x_offset=0.12):

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Position further to the right of the x-axis
    x = x_max + x_offset * x_range
    y = y_min

    # Scale bars
    ax.hlines(
        y, x, x + x_scale,
        color="black", lw=2,
        clip_on=False
    )

    ax.vlines(
        x, y, y + y_scale,
        color="black", lw=2,
        clip_on=False
    )

    # X label
    ax.text(
        x + x_scale / 2,
        y - 0.03 * y_range,
        f"{x_scale} s",
        ha="center",
        va="top",
        clip_on=False
    )

    # Y label
    ax.text(
        x - 0.02 * x_range,
        y + y_scale / 2,
        f"{y_scale} ΔF/F",
        ha="right",
        va="center",
        rotation=90,
        clip_on=False
    )

def find_shared_peaks(
    df,
    duration_df,
    FOV_id=None,
    neurons=None,
    fs=30.89,
    tolerance=10,
    peak_detect_type="peak_onset",
    filter_by_state=None,
    filter_by_distance=None,
):

    shared_window_ms = (tolerance / fs) * 1000

    if filter_by_state is not None:
        df = df[df.peak_state == filter_by_state]
        duration_df = duration_df[duration_df.state_mask==filter_by_state]

    if FOV_id is not None:
        df = df.xs(FOV_id, level="FOV_id").reset_index()
        df["FOV_id"] = FOV_id
        duration_df = duration_df[duration_df.FOV_id==FOV_id]

    grouped = df.groupby(["mouse_id", "date", "FOV_id"])

    results = []
    shared_peak_index = []

    for (mouse_id, date, FOV_id), fov_df in grouped:

        if neurons is None:
            fov_neurons = fov_df["neuron_id"].dropna().unique()
        else:
            fov_neurons = neurons

        # Only keep neurons actually present in this FOV
        fov_neurons = [
            n for n in fov_neurons
            if n in fov_df["neuron_id"].values
        ]

        if len(fov_neurons) < 2:
            continue

        for neuron_a, neuron_b in it.combinations(fov_neurons, 2):

            df_a = fov_df[fov_df["neuron_id"] == neuron_a]
            df_b = fov_df[fov_df["neuron_id"] == neuron_b]

            if df_a.empty or df_b.empty:
                continue

            # -----------------------------------
            # Distance
            # -----------------------------------

            xy_dist = np.sqrt(
                (df_a.x_coord.iloc[0] - df_b.x_coord.iloc[0]) ** 2
                + (df_a.y_coord.iloc[0] - df_b.y_coord.iloc[0]) ** 2
            )
            xy_dist_rounded = (np.round(xy_dist / 30) * 30)

            if filter_by_distance is not None:
                if xy_dist_rounded > filter_by_distance:
                    continue

            # -----------------------------------
            # Peak extraction
            # -----------------------------------

            df_a_sorted = (
                df_a
                .dropna(subset=[peak_detect_type])
                .sort_values(peak_detect_type)
            )

            df_b_sorted = (
                df_b
                .dropna(subset=[peak_detect_type])
                .sort_values(peak_detect_type)
            )

            peaks_a = (
                df_a_sorted[peak_detect_type]
                .astype(int)
                .to_numpy()
            )

            peaks_b = (
                df_b_sorted[peak_detect_type]
                .astype(int)
                .to_numpy()
            )

            # Skip if either neuron has less than 10 detected peaks
            if len(peaks_a) == 0 or len(peaks_b) == 0:
                continue

            # -----------------------------------
            # Preserve neuron type
            # -----------------------------------

            lineage_a = df_a["neuron_type"].iloc[0]
            lineage_b = df_b["neuron_type"].iloc[0]

            # -----------------------------------
            # Assign lineage label
            # -----------------------------------

            if lineage_a == "OP-der" and lineage_b == "OP-der":
                lineage_label = "OP-OP"

            elif lineage_a == "IP-der" and lineage_b == "IP-der":
                lineage_label = "IP-IP"

            elif {lineage_a, lineage_b} == {"IP-der", "OP-der"}:
                lineage_label = "IP-OP"

            else:
                lineage_label = "Other"

            # -----------------------------------
            # One-to-one peak matching
            # -----------------------------------

            # Matrix of absolute temporal differences
            distance_matrix = np.abs(
                peaks_a[:, None] - peaks_b[None, :]
            )

            # Find peaks where distance matrix < tolerance
            # Keep every pair within tolerance
            a_idx, b_idx = np.where(distance_matrix <= tolerance)

            # Matched peak locations
            matched_peaks_a = peaks_a[a_idx]
            matched_peaks_b = peaks_b[b_idx]

            # Signed timing difference
            # Positive = B occurred after A
            # Negative = B occurred before A
            peak_differences = (
                matched_peaks_b - matched_peaks_a
            )

            # Absolute difference
            peak_differences_abs = np.abs(
                peak_differences
            )

            # -----------------------------------
            # Match counts / percentage
            # -----------------------------------

            matched_peaks = len(a_idx) 

            total_peaks = (len(peaks_a) + len(peaks_b)) / 2

            percentage = (
                100 * matched_peaks / total_peaks
                if total_peaks > 0
                else np.nan
            )

            # -----------------------------------
            # find number of events expected by chance
            n_a = len(peaks_a)
            n_b = len(peaks_b)

            duration = duration_df[duration_df.FOV_id==FOV_id].state_duration.iloc[0]

            rate_a = n_a / duration
            rate_b = n_b / duration

            window = tolerance / fs

            expected_shared = (
                2
                * window
                * rate_a
                * rate_b
                * duration
            )

            # -----------------------------------
            # Results
            # -----------------------------------

            results.append({
                "mouse_id": mouse_id,
                "date": date,
                "FOV_id": FOV_id,

                "neuron_a": neuron_a,
                "neuron_b": neuron_b,

                "shared_window_ms": shared_window_ms,
                "duration":duration,

                "neuron_a_uid": f"{FOV_id}_n{neuron_a}",
                "neuron_type_a": lineage_a,

                "neuron_b_uid": f"{FOV_id}_n{neuron_b}",
                "neuron_type_b": lineage_b,

                "dff_peaks_a":np.max(df_a_sorted['peak_height']),
                "dff_peaks_b":np.max(df_b_sorted['peak_height']),

                "mean_peak_amp":np.mean([df_a_sorted['peak_height'].mean(),df_b_sorted['peak_height'].mean()]),
                "mean_event_freq":np.mean([n_a,n_b]) / (duration / 60), 

                "lineage_label": lineage_label,
                "XY_distance": xy_dist_rounded,
                "XY_dist_raw":xy_dist,
                "matched_peaks": matched_peaks,
                "percent_shared": percentage,
                "expected_shared":expected_shared,
                #'events_above_chance':(matched_peaks - expected_shared) / (total_peaks / (duration / 60)) # duration in minutes
                'events_above_chance':(matched_peaks - expected_shared)  / (duration / 60) # duration in minutes
            })

            # -----------------------------------
            # Store individual matched peaks
            # -----------------------------------

            shared_peak_states_a = (
                df_a_sorted.iloc[a_idx]["peak_state"]
                .to_numpy()
            )

            shared_peak_index.append({
                "FOV_id": FOV_id,
                "shared_window_ms": shared_window_ms,

                "neuron_a": neuron_a,
                "neuron_b": neuron_b,

                "idx_a": matched_peaks_a,
                "idx_b": matched_peaks_b,

                "peak_difference_abs_samples":
                    peak_differences_abs,

                "peak_difference_abs_ms":
                    peak_differences_abs / fs * 1000,

                "shared_peak_type":
                    shared_peak_states_a,
            })

    # -----------------------------------
    # Format shared peak index dataframe
    # -----------------------------------

    shared_peak_indices = pd.DataFrame(
        shared_peak_index
    )

    if not shared_peak_indices.empty:

        shared_peak_indices = (
            shared_peak_indices
            .explode(
                [
                    "idx_a",
                    "idx_b",
                    "peak_difference_abs_samples",
                    "peak_difference_abs_ms",
                    "shared_peak_type",
                ],
                ignore_index=True
            )
        )

        shared_peak_indices.dropna(
            subset=["idx_a"],
            inplace=True
        )

    return (
        pd.DataFrame(results),
        shared_peak_indices
    )

def add_example_trace(ax0,ax1,example):

    datapath = Path('/Volumes/MAC_FORMAT/Data_postdoc/Josh in vivo/processed')
    fs = 30.89  # in Hz
    ymax = 6

    # IP-OP example: 
    # 5 events in each, 1 shared event, 52s, dFF ~6
    if example == 'IP-OP':
        FOV_id = '2025_12_02_CBLK1926_1b_FOV6'
        neuron_id1 = 9
        neuron_id2 = 12
        x1, x2 = [430,480]

    if example == 'IP-IP':
        # IP-IP example: 
        # 7-10 events in each, 2 shared events, 45s dFF ~ 6/7
        FOV_id = '2025_11_27_CBLK1926_1d_FOV1'
        neuron_id1 = 8
        neuron_id2 = 25
        x1, x2 = [447,497]
        #x1,x2 = [40,90]

    # OP-OP examples: 
    # 
    # FOV_id = '2025_12_03_CBLK1926_1b_FOV1'
    # neuron_id1 = 3
    # neuron_id2 = 7
    # x1, x2 = [480,540]

    # FOV_id = '2025_12_03_CBLK1926_1b_FOV1'
    # neuron_id1 = 1
    # neuron_id2 = 6
    # x1, x2 = [450,510]

    # FOV_id = '2025_12_02_CBLK1926_1b_FOV3'
    # neuron_id1 = 0
    # neuron_id2 = 16
    # x1, x2 = [600,660]

    if example == 'OP-OP':
        # 6ish events in each, dFF 5, 60s
        FOV_id = '2025_12_02_CBLK1926_1b_FOV6'
        neuron_id1 = 9
        neuron_id2 = 13
        x1, x2 = [80,130]


    neurons = [neuron_id1,neuron_id2]
    trace_type = 'dff_smoothed'
    detect_type = 'peak_onset' # index or onset 
    # only load in the example neurons
    calcium = pd.read_parquet(datapath / "calcium_signals.parquet", 
                                            columns=['x_coord',
                                                    'y_coord',
                                                    trace_type,
                                                    'state_mask'],
                                            filters=[
                                                    ("FOV_id", "==", FOV_id),
                                                    ("neuron_id", "in", neurons),
                                                ]
                                                    )

    csv_path = Path('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1')
    calcium_events = pd.read_csv(csv_path / 'all_calcium_events.csv', index_col=['FOV_id','neuron_id','neuron_type'])
    duration_df = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 1/durations_by_state.csv')

    shared_peaks, shared_peak_idx = find_shared_peaks(calcium_events,
                                                    duration_df,
                                                    FOV_id=FOV_id,
                                                    neurons=neurons,
                                                    tolerance=30,
                                                    peak_detect_type=detect_type,
                                                    filter_by_state='spontaneous',
                                                    )

    calcium_events1 = calcium_events.xs((FOV_id,neuron_id1),level=('FOV_id','neuron_id'))[detect_type]
    calcium_events2 = calcium_events.xs((FOV_id,neuron_id2),level=('FOV_id','neuron_id'))[detect_type]

    calcium1 = calcium.xs((FOV_id,neuron_id1),level=('FOV_id','neuron_id'))[trace_type]
    calcium2 = calcium.xs((FOV_id,neuron_id2),level=('FOV_id','neuron_id'))[trace_type]

    calcium1_state_mask = calcium.xs((FOV_id,neuron_id1),level=('FOV_id','neuron_id'))['state_mask']

    time = calcium1.index.get_level_values('time')

    ax0.plot(time,calcium1-np.nanmedian(calcium1),color=colors[shared_peaks.neuron_type_a.iloc[0]],alpha=.4)
    ax1.plot(time,calcium2-np.nanmedian(calcium2),color=colors[shared_peaks.neuron_type_b.iloc[0]],alpha=.4)

    # option to plot state mask over the top of plot
    # states= calcium1_state_mask.unique()
    # states = ['spontaneous']
    # for state in states:
    #     ax[0].fill_between(
    #         time,
    #         0,
    #         ymax,
    #         where=(calcium1_state_mask== state),
    #         alpha=0.2,
    #         label=state
    #     )
    #     ax[1].fill_between(
    #         time,
    #         0,
    #         ymax,
    #         where=(calcium1_state_mask== state),
    #         alpha=0.2,
    #         label=state
    #     )

    ax0.vlines(
        time[calcium_events1.values],
        ymin=0,
        ymax=ymax,
        label=f'{shared_peaks.neuron_type_a.iloc[0]}ived events',
        color=colors[shared_peaks.neuron_type_a.iloc[0]],
        alpha=1,
        linestyle='dotted'
    )
    ax0.vlines(
        time[shared_peak_idx.idx_a.to_numpy(dtype=int)],
        ymin=0,
        ymax=ymax,
        label='Co-events',
        color=colors[shared_peaks.neuron_type_a.iloc[0]],
        alpha=1
    )

    ax1.vlines(
        time[calcium_events2.values],
        ymin=0,
        ymax=ymax,
        label=f'{shared_peaks.neuron_type_b.iloc[0]}ived events',
        color=colors[shared_peaks.neuron_type_b.iloc[0]],
        alpha=1,
        linestyle='dotted'
    )
    ax1.vlines(
        time[shared_peak_idx.idx_b.to_numpy(dtype=int)],
        ymin=0,
        ymax=ymax,
        label='Co-events',
        color=colors[shared_peaks.neuron_type_b.iloc[0]],
        alpha=1
    )

    ax0.set_xlim(x1,x2)
    ax1.set_xlim(x1,x2)

    ax0.axis('off')
    ax1.axis('off')

    ax0.legend()
    ax1.legend()

    if example=='IP-IP':
        scale_bar(ax1)

    ax0.set_title(example)

def compile_legends(axes,fig,y_pos=0.5):# Collect legend entries from multiple subplots
    handles = []
    labels = []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicate labels while preserving order
    unique = dict(zip(labels, handles))
    labels = list(unique.keys())
    handles = list(unique.values())

    # Remove individual subplot legends
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # One combined legend
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, y_pos),
        ncol=len(labels),
        frameon=False,
        prop={"family": "Arial", "size": 8}
    )