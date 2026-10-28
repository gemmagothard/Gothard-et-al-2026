import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code_utils import analysis_plotting as pl
import numpy as np
import figure1_utils as f1_ut
from matplotlib.patches import Circle
import f6_utils as f6_ut
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.gridspec import GridSpec


colors = {
        'OP-der':'#00CDE5',
        'IP-der':'#E404FF'}

palette_bars = {
                'OP-OP':'#00CDE5',
                'IP-OP': '#AE7DF8',
                'IP-IP':'#E404FF',
                'Distance-only null':'grey'
                }

def generate_3d_kde(kde_2d, n_samples=10000, bw=0.5, seed=42):
    rng = np.random.default_rng(seed)

    # Sample XY coordinates from 2D KDE
    xy = kde_2d.resample(n_samples, seed=rng).T

    # Treat lateral X as radial distance
    r = np.abs(xy[:, 0])
    y = xy[:, 1]

    # Rotate around Y-axis
    theta = rng.uniform(0, 2 * np.pi, n_samples)

    x = r * np.cos(theta)
    z = r * np.sin(theta)

    xyz = np.column_stack([x, y, z])

    # Fit 3D KDE
    kde_3d = gaussian_kde(xyz.T, bw_method=bw)

    return kde_3d, xyz


def resample_kde(kde, n_samples=100, seed=None):
    # Sample new PV positions from original KDE
    sampled_xy = kde.resample(n_samples, seed=seed).T

    # Fit a new KDE to sampled positions
    new_kde = gaussian_kde(sampled_xy.T)

    return new_kde

def monte_carlo_effect(kde_op, kde_ip, pv_candidates, n_sim=1000):

    means = {}

    for pair, kde_a, kde_b, id_a, id_b in [
        ('OP-OP', kde_op, kde_op, 'OP-der', 'OP-der'),
        ('IP-OP', kde_ip, kde_op, 'IP-der', 'OP-der'),
        ('IP-IP', kde_ip, kde_ip, 'IP-der', 'IP-der')
    ]:
        shared = simulate_shared_inputs(
            pv_candidates,
            kde_a,
            kde_b,
            id_a,
            id_b,
            distance=60,
            n_inputs=20,
            n_sim=n_sim,
            seed=42,
            show=False
        )

        means[pair] = np.nanmean(shared)

    return np.var(list(means.values())), means

def calculate_kde_sharing(
    kde_a,
    kde_b,
    distance=50,
    pial_depth_a=None,
    pial_depth_b=None,
    x_range=(-1000, 1000),
    y_range=(-500, 1000),
    resolution=300,
    show=False,
    ax=None
):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)

    X, Y = np.meshgrid(x, y)

    depth_difference = pial_depth_b-pial_depth_a

    positions_a = np.vstack([X.ravel(), Y.ravel()])
    positions_b = np.vstack([(X - distance).ravel(), Y.ravel()- depth_difference])

    f_a = kde_a(positions_a).reshape(X.shape)
    f_b = kde_b(positions_b).reshape(X.shape)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    overlap = np.sum(f_a * f_b) * dx * dy

    norm_a = np.sum(f_a**2) * dx * dy
    norm_b = np.sum(f_b**2) * dx * dy

    overlap = np.sum(np.minimum(f_a, f_b)) * dx * dy

    overlap_percent = 100 * overlap

    if show:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        # Overlay KDE contours
        ax.contourf(
            X, Y, f_a,
            levels=100,
            cmap='Blues',
            alpha=0.5
        )

        ax.contourf(
            X, Y, f_b,
            levels=100,
            cmap='Oranges',
            alpha=0.5
        )

        # Pyramidal neuron positions
        ax.scatter(
            [0, distance],
            [0, depth_difference],
            marker='^',
            s=100,
            c=['tab:blue', 'tab:orange'],
            edgecolors='black',
            zorder=10
        )

        ax.set(
            xlabel='X (µm)',
            ylabel='Y (µm)',
            title=f'KDE overlap at {distance} µm\n'
                  f'Product overlap = {overlap:.2e} µm⁻²'
        )

        ax.invert_yaxis()
        ax.set_aspect('equal')
        sns.despine(ax=ax)
        plt.show()

    return overlap_percent

def calculate_pv_density(pv_coords,show=False):

    xy = pv_coords[['X_rel', 'Y_rel']].dropna().to_numpy()
    
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    # Area in mm²
    area_mm2 = (
        (x_max - x_min) *
        (y_max - y_min)
    ) / 1e6

    # Mean PV density
    mean_density = len(xy) / area_mm2

    return mean_density

def generate_pv_candidates(
    pv_density=100,
    x_range=(-1000, 1000),
    y_range=(-200, 1000),
    kde=None,
    seed=42,
    show=False
):
    rng = np.random.default_rng(seed)

    # Area in mm²
    area_mm2 = (
        (x_range[1] - x_range[0]) *
        (y_range[1] - y_range[0])
    ) / 1e6

    n_candidates = round(pv_density * area_mm2)

    if kde is None:
        # Uniform sampling
        pv_candidates = rng.uniform(
            low=[x_range[0], y_range[0]],
            high=[x_range[1], y_range[1]],
            size=(n_candidates, 2)
        )

    else:
        # KDE sampling within spatial bounds
        samples = []

        while len(samples) < n_candidates:
            batch = kde.resample(n_candidates, seed=rng).T

            valid = batch[
                (batch[:, 0] >= x_range[0]) &
                (batch[:, 0] <= x_range[1]) &
                (batch[:, 1] >= y_range[0]) &
                (batch[:, 1] <= y_range[1])
            ]

            samples.extend(valid.tolist())

        pv_candidates = np.array(samples[:n_candidates])

    if show:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(
            pv_candidates[:, 0],
            pv_candidates[:, 1],
            s=15,
            color='black',
            alpha=0.6
        )

        ax.set(
            xlabel='X (µm)',
            ylabel='Y (µm)',
            title=f'Candidate PV interneurons (n={n_candidates})',
            xlim=x_range,
            ylim=y_range
        )
        ax.invert_yaxis()
        ax.set_aspect('equal')
        sns.despine()
        plt.show()

    return pv_candidates

def simulate_shared_inputs_null(
    pv_xy,
    distance=50,
    decay=124,
    n_inputs=20,
    n_sim=10000,
    seed=42,
    show=False,
    ax=None
):
    rng = np.random.default_rng(seed)

    neuron_a = np.array([0, 0])
    neuron_b = np.array([distance, 0])

    # Distance to each pyramidal neuron
    d_a = np.linalg.norm(pv_xy - neuron_a, axis=1)
    d_b = np.linalg.norm(pv_xy - neuron_b, axis=1)

    # Distance-only connectivity weights
    density_a = np.exp(-d_a / decay)
    density_b = np.exp(-d_b / decay)

    # Normalize to equal expected PV input counts
    def normalize_probabilities(weights, n_inputs):
        if not 0 <= n_inputs <= len(weights):
            raise ValueError("n_inputs must be between 0 and candidate count")

        low, high = 0.0, 1.0

        while np.minimum(high * weights, 1).sum() < n_inputs:
            high *= 2

        for _ in range(60):
            mid = (low + high) / 2

            if np.minimum(mid * weights, 1).sum() < n_inputs:
                low = mid
            else:
                high = mid

        return np.minimum(high * weights, 1)

    p_a = normalize_probabilities(density_a, n_inputs)
    p_b = normalize_probabilities(density_b, n_inputs)

    # Independently sample PV connections
    conn_a = rng.random((n_sim, len(pv_xy))) < p_a
    conn_b = rng.random((n_sim, len(pv_xy))) < p_b

    # Calculate percentage of shared PV inputs
    shared = (conn_a & conn_b).sum(axis=1)

    n_a = conn_a.sum(axis=1)
    n_b = conn_b.sum(axis=1)

    denominator = (n_a + n_b) / 2

    percentage = np.divide(
        100 * shared,
        denominator,
        out=np.full(n_sim, np.nan),
        where=denominator > 0
    )

    if show:
        ax_given = True

        if ax is None:
            ax_given = False
            fig, ax = plt.subplots(figsize=(6, 6))


        a = conn_a[0]
        b = conn_b[0]

        for mask, color, label in [
            (~a & ~b, 'lightgrey', 'Neither'),
            (a & ~b, 'tab:blue', 'A only'),
            (~a & b, 'tab:orange', 'B only'),
            (a & b, 'tab:green', 'Shared')
        ]:
            ax.scatter(
                pv_xy[mask, 0],
                pv_xy[mask, 1],
                s=15,
                color=color,
                label=f'{label} (n={mask.sum()})',
                alpha=0.8
            )

        ax.scatter(
            [0, distance],
            [0, 0],
            marker='^',
            s=40,
            c=['tab:blue', 'tab:orange'],
            edgecolors=['black', 'grey'],
            zorder=10
        )

        ax.set(
            xlabel='X position (µm)',
            ylabel='Y position (µm)',
            title=f'Distance-only null model | Shared = {percentage[0]:.1f}%'
        )
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=1
        )

        if not ax_given:
            plt.show()

    return percentage

def simulate_shared_inputs(
    pv_xyz,
    kde_a,
    kde_b,
    neurona_id,
    neuronb_id,
    distance=60,
    pial_depth_a=None,
    pial_depth_b=None,
    n_inputs = 10,
    n_sim=10000,
    seed=42,
    show=False,
    ax=None
):
    rng = np.random.default_rng(seed)

    # Default to original behaviour if depths aren't provided
    if pial_depth_a is None:
        pial_depth_a = 0

    if pial_depth_b is None:
        pial_depth_b = 0

    neuron_a = np.array([0, 0, 0])
    # second neuron is relative to first neuron
    neuron_b = np.array([distance, pial_depth_b-pial_depth_a, 0])

    # Evaluate lineage-specific KDEs
    relative_a = (pv_xyz - neuron_a).T
    relative_b = (pv_xyz - neuron_b).T

    # Presynaptic PV density
    density_a = kde_a(relative_a)
    density_b = kde_b(relative_b)

    def normalize_probabilities(density, n_inputs):
        """
        Convert KDE densities to connection probabilities,
        ensuring the expected number of PV inputs equals n_inputs.
        """
        density = np.asarray(density, dtype=float)

        if not 0 <= n_inputs <= len(density):
            raise ValueError("n_inputs must be between 0 and number of PV candidates")

        if np.any(density < 0) or not np.all(np.isfinite(density)):
            raise ValueError("Density must contain finite, non-negative values")

        if n_inputs == 0:
            return np.zeros_like(density)

        if np.count_nonzero(density) < n_inputs:
            raise ValueError("Not enough candidates with positive density")

        low, high = 0.0, 1.0

        # Find upper scaling bound
        while np.minimum(high * density, 1).sum() < n_inputs:
            high *= 2

        # Binary search
        for _ in range(60):
            mid = (low + high) / 2

            if np.minimum(mid * density, 1).sum() < n_inputs:
                low = mid
            else:
                high = mid

        return np.minimum(high * density, 1)

    def calculate_shared_inputs(conn_a,conn_b):
    
        shared = (conn_a & conn_b).sum(axis=1)
        n_a = conn_a.sum(axis=1)
        n_b = conn_b.sum(axis=1)

        denominator = (n_a + n_b) / 2

        percentage = np.divide(
            100 * shared,
            denominator,
            out=np.full(n_sim, np.nan),
            where=denominator > 0
        )
        return percentage
    

    p_a = normalize_probabilities(density_a, n_inputs)
    p_b = normalize_probabilities(density_b, n_inputs)

    # Simulate connections
    conn_a = rng.random((n_sim, len(pv_xyz))) < p_a

    # no cross-lineage rule
    conn_b = rng.random((n_sim, len(pv_xyz))) < p_b
    percentage = calculate_shared_inputs(conn_a,conn_b)

    if show:

        n = min(5, n_sim - 1)
        ax_given = ax is not None

        if not ax_given:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

        a = conn_a[n]
        b = conn_b[n]

        plot_colors = colors.copy()

        if neurona_id == neuronb_id:
            plot_colors[neurona_id] = mcolors.to_rgba(
                plot_colors[neurona_id], alpha=0.5
            )

        for mask, color, label in [
            #(~a & ~b, 'lightgrey', 'Neither'),
            (a & b, 'black', 'Shared'),
            #(a & ~b, plot_colors[neurona_id], f'{neurona_id} inputs (n1)'),
            #(~a & b, plot_colors[neuronb_id], f'{neuronb_id} inputs (n2)'),
            
        ]:
            ax.scatter(
                pv_xyz[mask, 0], # x
                pv_xyz[mask, 2], # z
                pv_xyz[mask, 1], # y
                s=15,
                color=color,
                label=f'{label} (n={mask.sum()})',
                edgecolors='black' if label in [
                    f'{neurona_id} inputs (n1)', 'Shared'
                ] else 'none',
                alpha=0.7
            )

        # Plot pyramidal neurons
        ax.scatter(
            [neuron_a[0], neuron_b[0]],
            [neuron_a[2], neuron_b[2]],
            [neuron_a[1], neuron_b[1]],
            marker='^',
            s=100,
            c=[plot_colors[neurona_id], plot_colors[neuronb_id]],
            edgecolors=['black', 'none'],
            depthshade=False,
            zorder=10
        )

        ax.set(
            title=f'Simulated PV inputs'
        )

        # Invert cortical depth axis
        ax.set_xlim(-500,500)
        ax.set_zlim(-250,1000)
        ax.set_ylim(-170,170)

        ax.invert_zaxis()
        ax.set_aspect('equal')

        ax.legend(
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=2
        )

        if not ax_given:
            plt.show()

    return percentage


real_data = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 2/ipsc_results.csv')
PV_all = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/PV_input_results.csv')
PV_csv = PV_all[PV_all.id=='pv_inputs']

# Observed PV coordinates
op_pv_xy = PV_csv.loc[
    PV_csv.condition == 'OP-der', ['X_rel', 'Y_rel']
].dropna().to_numpy()

ip_pv_xy = PV_csv.loc[
    PV_csv.condition == 'IP-der', ['X_rel', 'Y_rel']
].dropna().to_numpy()

kde_op = gaussian_kde(op_pv_xy.T)
kde_ip = gaussian_kde(ip_pv_xy.T)
kde_null = gaussian_kde(np.vstack([op_pv_xy, ip_pv_xy]).T)

# Generate 3D KDEs
kde_op_3d, op_pv_xyz = generate_3d_kde(kde_op)
kde_ip_3d, ip_pv_xyz = generate_3d_kde(kde_ip)
kde_null_3d, null_pv_xyz = generate_3d_kde(kde_null)

max_density = max(
    kde_op_3d(op_pv_xyz.T).max(),
    kde_ip_3d(ip_pv_xyz.T).max()
)

# PV density based on literature
pv_density = 8194  # PV neurons/mm³, based on Palicz et al. 2024

slice_thickness_um = 350
volume_mm3 = (
    1300 * # cortical depth
    2000 * # cortical width
    slice_thickness_um # cortical thickness
) / 1e9

# Number of PV neurons based on measured density
n_candidates = round(pv_density * volume_mm3)

print(f"Mean PV density: {pv_density:.1f} cells/mm²")
print(f"Simulation area: {volume_mm3:.3f} mm3")
print(f"Expected PV neurons: {n_candidates}")

layer_depths = {'Layer 1':[-250,-150],
                'Layer 2/3':[-150,300],
                'Layer 4':[300,450],
                'Layer 5':[450,700],
                'Layer 6':[700,1000]
                }

layer_densities = {'Layer 1':0,
                   'Layer 2/3':1200,
                   'Layer 4':2100,
                   'Layer 5':1800,
                   'Layer 6':1000}

def generate_pv_candidates(
    pv_density=1500,  # cells/mm³
    x_range=(-1000, 1000),
    y_range=(-250, 1000),
    z_range=(-175, 175),
    seed=42
):
    if seed == 'random':
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    # Volume in mm³
    volume_mm3 = (
        (x_range[1] - x_range[0]) *
        (y_range[1] - y_range[0]) *
        (z_range[1] - z_range[0])
    ) / 1e9

    # Number of PV interneurons
    n_candidates = round(pv_density * volume_mm3)

    # Uniform random sampling in 3D
    x = rng.uniform(*x_range, n_candidates)
    y = rng.uniform(*y_range, n_candidates)
    z = rng.uniform(*z_range, n_candidates)

    return np.column_stack([x, y, z])

def estimate_pv_inputs(
    pv_density=8194,       # PV cells/mm³
    decay=124,             # Distance decay constant (µm)
    p0=0.8,                # Connection probability at 0 µm
    x_range=(-1000, 1000),
    y_range=(-250, 1000),
    z_range=(-175, 175),
    neuron_position=(0, 0, 0),
    n_sim=100,
    seed=42
):
    rng = np.random.default_rng(seed)

    # Calculate tissue volume (mm³)
    volume_mm3 = (
        (x_range[1] - x_range[0]) *
        (y_range[1] - y_range[0]) *
        (z_range[1] - z_range[0])
    ) / 1e9

    # Number of available PV interneurons
    n_candidates = round(pv_density * volume_mm3)

    expected_inputs = []

    for _ in range(n_sim):

        # Randomly sample PV positions in 3D
        pv_xyz = rng.uniform(
            low=[x_range[0], y_range[0], z_range[0]],
            high=[x_range[1], y_range[1], z_range[1]],
            size=(n_candidates, 3)
        )

        # Distance to pyramidal neuron
        distances = np.linalg.norm(
            pv_xyz - np.asarray(neuron_position),
            axis=1
        )

        # PV -> pyramidal connection probability
        p_connect = p0 * np.exp(-distances / decay)

        # Expected number of presynaptic PV cells
        expected_inputs.append(p_connect.sum())

    return np.mean(expected_inputs), np.std(expected_inputs)

pv_candidates = generate_pv_candidates(pv_density=pv_density)

# Estimate PV convergence
n_input_pvs, std_inputs = estimate_pv_inputs(
    pv_density=pv_density,
    decay=124,
    p0=1,
    n_sim=100
)

print(
    f"Estimated presynaptic PV cells: "
    f"{n_input_pvs:.1f} ± {std_inputs:.1f}"
)

n_candidates = len(pv_candidates)

chance_shared = 100 * n_input_pvs / n_candidates
print(f'chance of random sharing: {chance_shared}')

# Simulate shared inputs
intersomatic_distance = 60
n_sim = 5000

fig = plt.figure(figsize=(15, 6))
gs = GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[3,1]
    )
gs0 = gs[0].subgridspec(1, 3)
ax_op_op = fig.add_subplot(gs0[0, 0],projection='3d')
ax_ip_op = fig.add_subplot(gs0[0, 1],projection='3d')
ax_ip_ip = fig.add_subplot(gs0[0, 2],projection='3d')
ax_result = fig.add_subplot(gs[1])

results = []
for pair, neuron_ids, kde_a, kde_b, ax in [
    ('OP-OP', ['OP-der','OP-der'], kde_op_3d, kde_op_3d, ax_op_op),
    ('IP-OP', ['IP-der','OP-der'], kde_ip_3d, kde_op_3d, ax_ip_op),
    ('IP-IP', ['IP-der','IP-der'], kde_ip_3d, kde_ip_3d, ax_ip_ip),
    #('Null',  ['Unlabeled','Unlabeled'],kde_ip_3d, kde_null_3d, None),
    ]:

    real_pairs = real_data[real_data.pair_type==pair]

    show=True

    for _,row in real_pairs.iterrows():


        if pd.isna(row.neuron1_dist) or pd.isna(row.neuron2_dist):
            continue
    
        neuron1_id = neuron_ids[0]
        neuron2_id = neuron_ids[1]

        if neuron1_id == neuron2_id:
            dist1 = row.neuron1_dist
            dist2 = row.neuron2_dist
        elif row.neuron1_type == neuron1_id:
            dist1 = row.neuron1_dist
            dist2 = row.neuron2_dist
        else:
            dist1 = row.neuron2_dist
            dist2 = row.neuron1_dist

        shared = simulate_shared_inputs(
            pv_candidates,
            kde_a,
            kde_b,
            neurona_id=neuron_ids[0],
            neuronb_id=neuron_ids[1],
            distance=row.distance_apart,
            pial_depth_a=dist1,
            pial_depth_b=dist2,
            n_sim=n_sim,
            n_inputs=n_input_pvs,
            seed=42,
            show=show,
            ax=ax
        )

        results.append({'pair_type':pair,
                        'pair_id':row.pair_id,
                        'percent_shared':np.mean(shared),
                        'real_percent_shared':row.percent_shared_mean
        })
        show=False 

result_df = pd.DataFrame(results)
plot_df = result_df.melt(
    id_vars='pair_type',
    value_vars=['percent_shared', 'real_percent_shared'],
    var_name='source',
    value_name='relative_sharing'
)
plot_df['source'] = plot_df['source'].map({
    'percent_shared': 'Spatial simulation',
    'real_percent_shared': 'Experimental'
})

sns.barplot(
    data=plot_df,
    x='pair_type',
    y='relative_sharing',
    hue='source',
    hue_order=['Spatial simulation','Experimental'],
    order=['OP-OP', 'IP-OP', 'IP-IP'],
    errorbar='se',
    capsize=0,
    ax=ax_result
)
ax_result.legend(
    frameon=False,
    loc='upper center',
    bbox_to_anchor=(1.3, 0.5),
    ncol=1
)
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/simulated.png')
plt.show()







brain_labels = (
    PV_csv.groupby('brain_ID')['condition'].first()
)
    
# Observed means
observed_stat, observed_means = monte_carlo_effect(
    kde_op, 
    kde_ip, 
    pv_candidates, 
    n_sim=1000
)

observed_stat = np.var([
    observed_means['OP-OP'],
    observed_means['IP-OP'],
    observed_means['IP-IP']
])

# Store shuffled means
rng = np.random.default_rng(42)
null_means = []
null_var = []
n_perm = 500
for i in range(n_perm):

    # Your existing brain-level lineage shuffling
    shuffled_labels = rng.permutation(brain_labels.values)
    label_map = dict(zip(brain_labels.index, shuffled_labels))

    permuted = PV_csv.copy()
    permuted['condition'] = permuted['brain_ID'].map(label_map)

    op_xy = permuted.loc[
        permuted.condition == 'OP-der', ['X_rel', 'Y_rel']
    ].dropna().to_numpy()

    ip_xy = permuted.loc[
        permuted.condition == 'IP-der', ['X_rel', 'Y_rel']
    ].dropna().to_numpy()

    perm_kde_op = gaussian_kde(op_xy.T)
    perm_kde_ip = gaussian_kde(ip_xy.T)

    # Monte Carlo simulation
    var, means = monte_carlo_effect(
        perm_kde_op,
        perm_kde_ip,
        pv_candidates,
        n_sim=1000
    )

    null_var.append(var)
    null_means.append(means)

null_means_df = pd.DataFrame(null_means)


p_omnibus = (
    np.sum(null_var >= observed_stat) + 1
) / (len(null_var) + 1)

print(f"Omnibus permutation P = {p_omnibus:.4f}")

ci_lower, ci_upper = np.percentile(null_var, [2.5, 95])

fig, ax = plt.subplots(1,1)
sns.histplot(null_var, 
             bins=30, color='grey', ax=ax)
ax.set(
    xlabel='Variance between lineage-specific mean sharing (%)',
    ylabel='Number of permutations'
)
# Shade central 95% interval
ax.axvspan(
    ci_lower, ci_upper,
    color='grey',
    alpha=0.2,
    label='95% null interval'
)
ax.axvline(
    observed_stat,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Observed variance'
)
plt.show()

from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

comparisons = [
    ('OP-OP', 'IP-OP'),
    ('OP-OP', 'IP-IP'),
    ('IP-OP', 'IP-IP')
]

posthoc = []
for group_a, group_b in comparisons:

    # Observed difference
    observed_diff = (
        observed_means[group_a] -
        observed_means[group_b]
    )

    # Difference for each brain-level permutation
    null_diff = (
        null_means_df[group_a] -
        null_means_df[group_b]
    ).to_numpy()

    # Two-sided permutation test
    p = (
        np.sum(np.abs(null_diff) >= abs(observed_diff)) + 1
    ) / (len(null_diff) + 1)

    posthoc.append({
        'comparison': f'{group_a} vs {group_b}',
        'difference': observed_diff,
        'p': p
    })

posthoc = pd.DataFrame(posthoc)

# Holm correction for multiple comparisons
posthoc['p_adj'] = multipletests(
    posthoc['p'],
    method='holm'
)[1]

print(posthoc)


fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

for ax, (group_a, group_b) in zip(axes, comparisons):

    observed_diff = (
        observed_means[group_a] -
        observed_means[group_b]
    )

    null_diff = (
        null_means_df[group_a] -
        null_means_df[group_b]
    )

    sns.histplot(
        null_diff,
        bins=30,
        color='grey',
        ax=ax
    )

    ax.axvline(
        observed_diff,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Observed'
    )

    ax.axvline(0, color='black', linewidth=0.8)

    ax.set(
        title=f'{group_a} vs {group_b}',
        xlabel='Difference in predicted sharing (%)',
        ylabel='Permutations'
    )

    ax.legend(frameon=False)
    sns.despine(ax=ax)

plt.tight_layout()
plt.show()