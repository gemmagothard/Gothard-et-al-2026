import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from diptest import diptest
import seaborn as sns
from scipy.ndimage import maximum_filter
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

colors = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }


def null_dist_p_test(observed,null,side='upper'):

    if side=='upper':
        # Upper tail
        p_upper = (np.sum(null >= observed) + 1) / (len(null) + 1)
        print("Upper-tail p =", p_upper)
        return p_upper
        
    elif side=='lower':
        # Lower tail
        p_lower = (np.sum(null <= observed) + 1) / (len(null) + 1)
        print("Lower-tail p =", p_lower)
        return p_lower
    
    elif side=='both':
        p_upper = (np.sum(null >= observed) + 1) / (len(null) + 1)
        p_lower = (np.sum(null <= observed) + 1) / (len(null) + 1)
        # Two-sided p-value
        p = min(1.0, 2 * min(p_lower, p_upper))
        print("Two-sided modality p =", p)
        return p

def kde_multimodality_score(
    kde,
    bounds=(-4, 4, -4, 4),
    gridsize=200,
    neighbourhood=15,
    min_height=0.02,
    show=False,
    ax=None,
    color=None,
    title=None
):
    """
    Continuous measure of multimodal structure in a 2D KDE.

    Score = mode_height - all other mode heights

    """

    xmin, xmax, ymin, ymax = bounds

    x = np.linspace(xmin, xmax, gridsize)
    y = np.linspace(ymin, ymax, gridsize)

    X, Y = np.meshgrid(x, y)

    points = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    Z = kde(points).reshape(X.shape)

    # Find local maxima
    local_max = (
        Z == maximum_filter(
            Z,
            size=neighbourhood,
            mode="nearest"
        )
    )

    # Exclude boundaries
    local_max[[0, -1], :] = False
    local_max[:, [0, -1]] = False

    # Ignore extremely small peaks
    local_max &= (
        Z >= min_height * Z.max()
    )

    iy, ix = np.where(local_max)

    heights = Z[iy, ix]
    #heights = heights[heights>min_height]

    if len(heights) == 0:
        print('no modes found ')
        return np.nan

    # Highest first
    order = np.argsort(heights)[::-1]
    heights = heights[order]
    
    ix = ix[order]
    iy = iy[order]

    # multimodality score
    if len(heights)>0:
        score = 1 + np.sum(heights[1:] / heights[0])

    if show:

        created_ax = ax is None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        if color is None:
            color='blue'
        cmap = LinearSegmentedColormap.from_list("custom",["white", color])


        cf = ax.contourf(
            X, Y, Z,
            levels=100,
            cmap=cmap
        )

        scores = np.array([h / heights[0] for h in heights])
        # plot where the detected modes are
        sc = ax.scatter(
            x[ix],
            y[iy],
            marker=".",
            s=5 + 100 * scores**2,
            c=scores,  # Colour based on value at each mode
            cmap="cividis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(sc, ax=ax, label="Mode weighting",location='left')
        
        offset = 4
        ax.text(
                x[ix[0]], y[iy[0]]+offset,
                f"Modality score: {score:.2f}",
                fontsize=8,
                ha="center",
                va="bottom"
            )


        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(title)


        if created_ax:
            plt.show()

    return score

def central_kde_aspect_ratio(
    kde,
    xy,
    level=0.8,
    bounds=(-4, 4, -4, 4),
    grid_size=200,
    show=False,
    ax=None,
    color=None,
    title=None
):
    """
    Aspect ratio of central KDE contour, forcing the major axis
    to be horizontal (X).

    AR = horizontal width / vertical height
    """

    xmin, xmax, ymin, ymax = bounds

    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)

    X, Y = np.meshgrid(x, y)

    positions = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    Z = kde(positions).reshape(X.shape)

    # Main KDE mode
    iy, ix = np.unravel_index(np.argmax(Z), Z.shape)

    mode_xy = np.array([
        X[iy, ix],
        Y[iy, ix]
    ])

    # 60% of peak density
    contour_level = level * Z.max()

    # Extract contour
    fig_tmp, ax_tmp = plt.subplots()

    cs = ax_tmp.contour(
        X,
        Y,
        Z,
        levels=[contour_level]
    )

    plt.close(fig_tmp)

    paths = cs.get_paths()

    # Find contour containing main mode
    selected = None

    for path in paths:
        if path.contains_point(mode_xy):
            selected = path.vertices
            break

    if selected is None:
        return np.nan, mode_xy, None

    # ----------------------------------
    # Horizontal / vertical dimensions
    # ----------------------------------

    x_min = selected[:, 0].min()
    x_max = selected[:, 0].max()

    y_min = selected[:, 1].min()
    y_max = selected[:, 1].max()

    horizontal_width = x_max - x_min
    vertical_height = y_max - y_min

    aspect_ratio = horizontal_width / vertical_height

    # ----------------------------------
    # Show measurement
    # ----------------------------------

    if show:

        created_ax = ax is None

        if created_ax: 
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.figure

        if color is None:
            color='blue'
        cmap = LinearSegmentedColormap.from_list("custom",["white", color])

            
        levels = np.linspace(
            np.nanmin(Z),
            np.nanmax(Z),
            100
        )

        # Entire KDE in grey
        ax.contourf(
            X, Y, Z,
            levels=levels,
            cmap=cmap
        )

        # Mask values below threshold
        Z_colour = np.ma.masked_where(Z < contour_level,Z)

        #if len(levels_high) >= 2:
        cf = ax.contourf(
            X,
            Y,
            Z_colour,
            levels=levels,
            cmap=cmap,
        )
        # Shade everything below selected density threshold grey


        # Selected contour
        ax.plot(
            selected[:, 0],
            selected[:, 1],
            linewidth=1,
            label=f"{level:.0%} contour",
            color='grey'
        )

        # Horizontal width
        y_mid = (y_min + y_max) / 2
        ax.plot(
            [x_min, x_max],
            [y_mid, y_mid],
            linewidth=2,
            label=f"Width = {horizontal_width:.2f}",
            color='black'
        )

        # Vertical height
        x_mid = (x_min + x_max) / 2
        ax.plot(
            [x_mid, x_mid],
            [y_min, y_max],
            linewidth=2,
            label=f"Height = {vertical_height:.2f}",
            color='black'
        )

        offset=3
        ax.text(x_mid,
                y_max+offset,
                f"Aspect ratio: {aspect_ratio:.2f}",
                fontsize=8,
                ha='center',
                va='bottom'
        )

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        if created_ax:
            plt.show()

    return aspect_ratio, mode_xy, selected

def generate_kde(data,xcol,ycol):

    XY = data[[xcol, ycol]].to_numpy()

    XY_norm = (XY - XY.mean(axis=0)) / XY.std(axis=0)
    kde = gaussian_kde(XY_norm.T)

    return kde, XY_norm


if __name__ == "__main__":

    df = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/PV_input_results.csv')

    # ============================================================
    # SETTINGS
    # ============================================================

    x_col = "X_rel"       # <-- CHANGE
    y_col = "Y"       # <-- CHANGE

    condition_col = "condition"

    op_name = "OP-der"
    ip_name = "IP-der"
    
    op = (
        df[df[condition_col] == op_name]
        .dropna(subset=[x_col, y_col])
        .copy())

    ip = (
        df[df[condition_col] == ip_name]
        .dropna(subset=[x_col, y_col])
        .copy())

    OP_kde, OP_xy = generate_kde(op,x_col,y_col)
    IP_kde, IP_xy = generate_kde(ip,x_col,y_col)

    level = 0.8
    OP_AR,_,_ = central_kde_aspect_ratio(OP_kde, OP_xy,level=level)
    IP_AR, _,_ = central_kde_aspect_ratio(IP_kde, IP_xy,level=level)
    op_modality_score = kde_multimodality_score(OP_kde,show=True)
    ip_modality_score = kde_multimodality_score(IP_kde,show=True)

    rng = np.random.default_rng(6)
    n_sim = 5000
    n_ip = len(IP_xy)
    simulation_results = []
    for i in range(n_sim):
        #print(f'simulation {i} / {n_sim}')
        # Generate IP-sized sample from OP KDE
        sim_xy = OP_kde.resample(n_ip,seed=rng).T

        # Apply same normalization as observed IP
        sim_xy = (sim_xy - sim_xy.mean(axis=0)) / sim_xy.std(axis=0)

        # Fit KDE to simulated sample
        sim_kde = gaussian_kde(sim_xy.T)

        # Calculate the aspect-ratio statistic
        aspect_ratio,_,_ = central_kde_aspect_ratio(sim_kde,sim_xy,level=level)
        # modality score
        modality_score = kde_multimodality_score(sim_kde)

        simulation_results.append({'simulation_n':i,
                                'aspect_ratio':aspect_ratio,
                                'modality_score':modality_score
                                })

    sim_results = pd.DataFrame(simulation_results)
    sim_results.to_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/simulated_results.csv')

    fig, (aspect_ax, mode_ax) = plt.subplots(1,2)

    observed = IP_AR
    null = sim_results.aspect_ratio.values
    p_val = null_dist_p_test(observed,null,side='both')
    aspect_ax.hist(null, bins=50, density=True,color='grey')
    aspect_ax.axvline(
        observed,
        linestyle="--",
        label="Observed IP",
        color=colors['IP-der'],
    )
    aspect_ax.axvline(
        OP_AR,
        linestyle="--",
        label="Observed OP",
        color=colors['OP-der'],
    )
    # 95% null boundaries
    aspect_ax.axvline(
        np.quantile(null, 0.025),
        linestyle=":",
        label=f"95% confidence interval",
        color='grey')
    aspect_ax.axvline(
        np.quantile(null, 0.975),
        linestyle=":",
        color='grey')
    aspect_ax.set_ylabel("Density")
    aspect_ax.legend()
    aspect_ax.set_xlabel('Aspect ratio')
    aspect_ax.set_title(f'p = {p_val:.3f}')

    observed = ip_modality_score
    null = sim_results.modality_score.values
    p_val = null_dist_p_test(observed,null,side='both')
    mode_ax.hist(null, bins=50, density=True,color='grey')
    mode_ax.axvline(
        observed,
        linestyle="--",
        label="Observed IP",
        color=colors['IP-der'],
    )
    mode_ax.axvline(
        op_modality_score,
        linestyle="--",
        label="Observed OP",
        color=colors['OP-der'],
    )
    # 95% null boundaries
    mode_ax.axvline(
        np.quantile(null, 0.975),
        linestyle=":",
        label=f"95% confidence interval",
        color='grey')
    mode_ax.axvline(
        np.quantile(null, 0.025),
        linestyle=":",
        color='grey')
    mode_ax.set_ylabel("Density")
    mode_ax.legend()
    mode_ax.set_xlabel('Uni-modality score')
    mode_ax.set_title(f'p = {p_val:.3f}')

    sns.despine(fig,top=True,right=True)
    plt.show()