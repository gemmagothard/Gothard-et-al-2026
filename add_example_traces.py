import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from code_utils import analysis_plotting as pl


def scale_bar(
    ax,
    x_scale=0.1,
    y_scale=100,
    x_label="s",
    y_label="pA",
    x_offset=0.5,
    y_offset=0.1,
    lw=1.5,
    fontsize=5
):

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Outside right edge
    x = x_max + x_offset * x_range
    y = y_min + y_offset * y_range

    # Horizontal bar
    ax.plot(
        [x, x + x_scale], [y, y],
        color="black",
        lw=lw,
        clip_on=False,
        scalex=False,
        scaley=False
    )

    # Vertical bar
    ax.plot(
        [x, x], [y, y + y_scale],
        color="black",
        lw=lw,
        clip_on=False,
        scalex=False,
        scaley=False
    )

    # X label 
    ax.text(
        x ,
        y - y_offset * y_range,
        f"{x_scale:g} {x_label}",
        ha="left",
        va="top",
        fontsize=fontsize,
        clip_on=False
    )

    # Y label
    ax.text(
        x ,
        y ,
        f"{y_scale:g} {y_label}",
        ha="right",
        va="baseline",
        rotation=90,
        fontsize=fontsize,
        clip_on=False
    )



def add_example_to_plot(fig, subplot_spec, type, example=None,panel_letter=None):

    plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':8
    })

    #example = 'IP-IP'

    datapath = Path('/Volumes/Gemma_data/Data_dphil/in_vitro_spontaneous_data')

    if not datapath.is_dir():
        return

    colors = {
            'red':'#00CDE5',
            'green': '#CB23E0',
            'grey':'grey'
            }

# IPSC EXAMPLES
    # OP-OP EXAMPLES:
    # whole 346-364s - OP-OP example, folder: P 210323_001, max = 225
    # 375 to 390
    # shared response 1 time: 350.95 - 351.45, ymax = 
    # shared response 2 time: 354.75 - 355.25, ymax = 225

    # IP-IP EXAMPLES:
    # folder: P 210218_002
    # whole: 165-183s, y_max = 175, 
    # shared response 1: 166.1-166.6
    # unshared response 1: 177-177.5
    # unshared response 2: 182.4-182.9

    #whole  260-290s, - aIP-OP example P 210322_003
    # second version: 270-290 // or 270-295
    # unshared input aIP2: 298.33-298.83
    # unshared input aIP3: 281.45-281.95

    # GREY EXAMPLES:
    # Grey-Grey example, folder: P 230202_008
    # 30-55 whole, y_max = 279
    # shared input1: 36.14-36.64
    # shared input2: 38.55-39.05
    # unshared input1: 41.82-41.32
    # unshared input2: 

# EPSC EXAMPLES
    # OP-OP: 210214_006
    if example == 'unlabeled':
        if type == 'ipsc':
            cell_id = 'P 230202_008'
            x1 = 30
            x2 = 50
            t1, t2 = 36.300, 36.800
            t3, t4 = 38.690, 39.190
            t5, t6 = 42.030, 42.530
            ex1_title = 'Shared'
            ex2_title = 'Shared'
            ex3_title = 'Unshared'


    if example=='IP-OP':
        if type == 'ipsc':
            cell_id = 'P 210322_003'
            x1 = 272
            x2 = 292
            t1, t2 = 277.770, 278.270
            t3, t4 = 282.815, 283.315
            t5, t6 = 285.260, 285.760
            ex1_title = 'Unshared'
            ex2_title = 'Unshared'
            ex3_title = 'Shared'
        if type == 'epsc':
            #cell_id = 'P 210218_004'
            #x1, x2 = 210, 230
            #t1,t2 = 210.94, 211.34
            #t3, t4 = 216.69, 217.09
            #t5, t6 = 229.45, 229.85

            cell_id = 'P 210214_001' # 170-190 but shared response is a bit big
            x1, x2 = 165, 185
            t1, t2 = 172.41, 172.71
            t3, t4 = 176.80, 177.10
            t5, t6 = 179.09, 179.39
            ex1_title = 'Unshared'
            ex2_title = 'Shared'
            ex3_title = 'Unshared'

            #cell_id = 'P 210221_001'
            #cell_id = 'P 210222_005'

    if example=='IP-IP':
        if type == 'ipsc':
            cell_id = 'P 210218_002'
            x1, x2 = 165, 185
            t1, t2 = 166.9, 167.4
            t3, t4 = 177.85, 178.35
            t5, t6 = 183.3, 183.8
            ex1_title = 'Shared'
            ex2_title = 'Unshared'
            ex3_title = 'Unshared'
        elif type == 'epsc':
            #cell_id = 'P 210322_001'
            cell_id = 'P 210214_003'
            #cell_id = 'P 210222_002'
            x1, x2 = 330, 350
            t1, t2 = 332.84, 333.14
            t3, t4 = 335.63, 335.93
            t5, t6 = 339.75, 340.05
            ex1_title = 'Unshared'
            ex2_title = 'Shared'
            ex3_title = 'Unshared'

    if example=='OP-OP':
        if type == 'ipsc':
            cell_id = "P 210323_001"
            x1 = 345
            x2 = 365
            t1, t2 = 352.65, 353.15
            t3, t4 = 354.25, 354.7
            t5, t6 = 356.45, 356.95
            ex1_title = 'Shared'
            ex2_title = 'Unshared'
            ex3_title = 'Shared'
        elif type == 'epsc':
            cell_id = 'P 210214_006'
            x1 = 342
            x2 = 362
            t1, t2 = 345.31, 345.61
            t3, t4 = 346.70, 347.00
            t5, t6 = 358.72, 359.02
            ex1_title = 'Unshared'
            ex2_title = 'Shared'
            ex3_title = 'Unshared'

    files = [
        f for f in datapath.rglob("*")
        if f.is_file()
        and type in f.name.lower()
        and f.suffix.lower() == ".abf"
        and (example is None or f.parent.name == cell_id)
    ]
    fs = 4096

    trace1 = []
    trace2 = []

    for file in files:

        df = pd.read_hdf(file,key='hdf')
        if len(df) == 0:
            print(f"Skipping empty dataframe: {file}")
            continue

        trace1_concat = np.concatenate([trace[4096:] for trace in df["Trace1"]])
        trace2_concat = np.concatenate([trace[4096:] for trace in df["Trace2"]])

        trace1_concat = savgol_filter(
            trace1_concat,
            window_length=15,
            polyorder=2
        )

        trace2_concat = savgol_filter(
            trace2_concat,
            window_length=15,
            polyorder=2
        )

        trace1.append(trace1_concat)
        trace2.append(trace2_concat)

        n1_id = df.pipette1.iloc[0]
        n2_id = df.pipette2.iloc[0]

        if example==None:
            print(n1_id,n2_id)
            if n1_id == 'red' and n2_id == 'green':
                plt.plot(trace1_concat,alpha=.5)
                plt.plot(trace2_concat,alpha=.5)
                plt.title(f"{file.name} - {n1_id} vs {n2_id}")
                plt.show()


    
    trace1 = np.concatenate(trace1)
    trace2 = np.concatenate(trace2)
    timevec = np.arange(0,len(trace1),1)/fs

    gs = subplot_spec.subgridspec(
        5, 3,
        height_ratios=[1, 1, 0.3, 0.7, 0.7],
        hspace=0.2,
        wspace=0.2
    )

    # Full-width traces
    full1 = fig.add_subplot(gs[0, :])
    full2 = fig.add_subplot(gs[1, :], sharex=full1, sharey=full1)

    if panel_letter is not None:
        pl.add_panel_letter(full1,panel_letter,y=1.5)

    # Three examples, each with two rows
    ex1_p1 = fig.add_subplot(gs[3, 0])

    if type=='ipsc':
        ymin = -25
        ymax = 270
        ex1_p1.set_ylim(ymin,ymax)
    elif type == 'epsc':
        ymin = -50
        ymax = 25

    ex1_p2 = fig.add_subplot(gs[4, 0], sharex=ex1_p1,sharey=ex1_p1)

    ex2_p1 = fig.add_subplot(gs[3, 1], sharey=ex1_p1)
    ex2_p2 = fig.add_subplot(gs[4, 1], sharex=ex2_p1, sharey=ex1_p1)

    ex3_p1 = fig.add_subplot(gs[3, 2], sharey=ex1_p1)
    ex3_p2 = fig.add_subplot(gs[4, 2], sharex=ex3_p1, sharey=ex1_p1)

    full1.plot(timevec,trace1,color=colors[n1_id],linewidth=0.5)
    full2.plot(timevec,trace2,color=colors[n2_id],linewidth=0.5)
    full1.set_title(f"{example.upper()}  |  s{type.upper()}s")

    full1.set_xlim(x1,x2)
    full1.set_ylim(ymin,ymax)

    time_windows = [
        (t1, t2),
        (t3, t4),
        (t5, t6),
    ]

    for start, end in time_windows:
        full1.axvspan(
            start,
            end,
            color="grey",
            alpha=0.2,
            linewidth=0
        )
    for start, end in time_windows:
        full2.axvspan(
            start,
            end,
            color="grey",
            alpha=0.2,
            linewidth=0
        )

    if example in ['IP-IP', 'unlabeled']:
        if type == 'ipsc':
            scale_bar(full2,x_scale=1,y_scale=100,x_label='s',y_label='pA',x_offset=0.05,y_offset=0.1,lw=1.5,fontsize=8)
        else:
            scale_bar(full2,x_scale=1,y_scale=50,x_label='s',y_label='pA',x_offset=0.05,y_offset=0.1,lw=1.5,fontsize=8)

    ex1_i1 = int(t1 * fs)
    ex1_i2 = int(t2 * fs)
    ex1_p1.plot(timevec[ex1_i1:ex1_i2]*1000,trace1[ex1_i1:ex1_i2],color=colors[n1_id],linewidth=0.8)
    ex1_p2.plot(timevec[ex1_i1:ex1_i2]*1000,trace2[ex1_i1:ex1_i2],color=colors[n2_id],linewidth=0.8)
    ex1_p1.set_title(ex1_title)

    ex1_i1 = int(t3 * fs)
    ex1_i2 = int(t4 * fs)
    ex2_p1.plot(timevec[ex1_i1:ex1_i2]*1000,trace1[ex1_i1:ex1_i2],color=colors[n1_id],linewidth=0.8)
    ex2_p2.plot(timevec[ex1_i1:ex1_i2]*1000,trace2[ex1_i1:ex1_i2],color=colors[n2_id],linewidth=0.8)
    ex2_p1.set_title(ex2_title)

    ex1_i1 = int(t5 * fs)
    ex1_i2 = int(t6 * fs)
    ex3_p1.plot(timevec[ex1_i1:ex1_i2]*1000,trace1[ex1_i1:ex1_i2],color=colors[n1_id],linewidth=0.8)
    ex3_p2.plot(timevec[ex1_i1:ex1_i2]*1000,trace2[ex1_i1:ex1_i2],color=colors[n2_id],linewidth=0.8)
    ex3_p1.set_title(ex3_title)


    for ax in [
        ex1_p1, ex1_p2,
        ex2_p1, ex2_p2,
        ex3_p1, ex3_p2
    ]:
        ax.add_patch(
            Rectangle(
                (0, 0),
                1, 1,
                transform=ax.transAxes,
                facecolor="0.9",
                edgecolor="none",
                zorder=-10,
                clip_on=True,
            )
        )

    if example == 'IP-IP':
        if type == 'ipsc':
            scale_bar(ex3_p2,x_scale=100,y_scale=100,x_label='ms',y_label='pA',x_offset=0.16,y_offset=0.1,lw=1.5,fontsize=8)
        elif type=='epsc':
            scale_bar(ex3_p2,x_scale=50,y_scale=50,x_label='ms',y_label='pA',x_offset=0.16,y_offset=0.1,lw=1.5,fontsize=8)

    panel_axes = [
        full1, full2,
        ex1_p1, ex1_p2,
        ex2_p1, ex2_p2,
        ex3_p1, ex3_p2
    ]

    for ax in panel_axes:
        ax.axis("off")


if __name__ == "__main__":

    fig=plt.figure()    
    gs = GridSpec(
        1, 1,
        figure=fig,
    )
    add_example_to_plot(fig,gs[0, 0],type='epsc',example='IP-OP')
    plt.show()