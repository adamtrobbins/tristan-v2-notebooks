# nohup python field_data_polished_cutaway_movie_script.py > cms.out 2>&1 &
# Input Settings
findex = -7
skip = 5
plot_choice = "density"
lazy = 2
contour_cheat = 2
buffer = 10
ybuffer = 450
lw = 2.3

var_to_plot = {'density': 'ndens', 'hall': 'ex'}[plot_choice]
cmap = {"density": "viridis", "hall": "seismic"}[plot_choice]
positive = {"density": True, "hall": False}[plot_choice]
vabsmax = {"density": 5.0, "hall": 0.1}[plot_choice]
ccolor = {"density": "white", "hall": "black"}[plot_choice]

split = 1
steps = range(0, 221, 1)

# Imports
from graphet import Data
from graphet.plugins import TristanV2 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import sys
import os
import time
import atexit
from multiprocessing import Pool, cpu_count

sys.path.append('../v2_proc_files')
from useful_funcs import *
from TristanRun import TristanRun
from FieldLineHelper import FieldLineHelper
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

plt.switch_backend('Agg')

# Initialization of global run data for workers
run = TristanRun(findex)
d = run.config_data(steps, "fields", includes_pressure = False)
cs_shift = np.load(f"{run.path}/crossings_data.npy")
cs_shift = np.concatenate((cs_shift, np.array([cs_shift[-1]])))
xpt_shift = np.load(f"{run.path}/xpt_data.npy")

def render_frame(rel_ts_idx):
    """
    Worker function to render a single frame. 
    By reusing the 'fig' object within the process, we optimize performance.
    """
    ts_val = steps[rel_ts_idx]
    print(f"Processing timestep {ts_val}...")
    
    # Logic to ensure each process creates its figure once and reuses it
    if not hasattr(render_frame, "fig"):
        render_frame.fig = plt.figure(figsize=(36, 12))
        render_frame.gs = GridSpec(2, 2, width_ratios=[1.5, 3], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        render_frame.flh1 = FieldLineHelper(run, 10, contour_ds=contour_cheat)
        render_frame.flh2 = FieldLineHelper(run, 8, contour_ds=contour_cheat)
        render_frame.flh3 = FieldLineHelper(run, 8, contour_ds=contour_cheat)
    else:
        render_frame.fig.clf() # Clear the figure for reuse

    fig = render_frame.fig
    gs = render_frame.gs
    
    title = f"$\Omega_i t = $ {round(rel_ts_idx * run.gyrofreq / run.mime):05d}"
    
    left_x_center = cs_shift[ts_val]
    right_x_center = cs_shift[ts_val]
    right_y_center = xpt_shift[ts_val]
    
    rects = [[left_x_center - 200, left_x_center + 200, 1200 - 300, 2000 - 300],
             [right_x_center - 120, right_x_center + 120, right_y_center - 240, right_y_center + 240]]
    colors = ['yellow', 'red']
    is_custom = [False, True]

    # Main Plotting Logic
    ax_main = fig.add_subplot(gs[0, :])
    roi_main = fq_cut(d.fields, rel_ts_idx, [0 + buffer, run.xlim - buffer, 0 + ybuffer, run.ylim - ybuffer])
    ez_rep = roi_main.ez.mean() * skip
    
    plot_fq3_flipped(roi_main, roi_main[var_to_plot], plot=ax_main, vabsmax=vabsmax, positive=positive, downsample=lazy, cmap=cmap)
    render_frame.flh1.plot_field_lines(roi_main, ez_rep, plot=ax_main, arrow_spacing=-1, linewidth=1, transpose=True, color=ccolor)
    
    ax_main.set_ylim(run.xlim - buffer, buffer)
    ax_main.set_title(title)

    # Zoom Plotting Logic
    for i, (rect, color) in enumerate(zip(rects, colors)):
        ax_zoom = fig.add_subplot(gs[1, i])
        [y1, y2, x1, x2] = rect
        roi_zoom = fq_cut(d.fields, rel_ts_idx, rect)
        
        plot_fq3_flipped(roi_zoom, roi_zoom[var_to_plot], plot=ax_zoom, vabsmax=vabsmax, positive=positive, downsample=lazy, cmap=cmap)
        
        curr_flh = render_frame.flh2 if i == 0 else render_frame.flh3
        curr_flh.plot_field_lines(roi_zoom, ez_rep, plot=ax_zoom, arrow_spacing=-1, transpose=True, color=ccolor)
        
        ax_zoom.tick_params(axis='both', which='both', direction='in', length=4)
        ax_zoom.set_xlabel("$y$ [$c/\\omega_{p}$]")
        ax_zoom.set_ylabel("$x$ [$c/\\omega_{p}$]")
        
        # Connection Lines
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=lw)
        ax_main.add_patch(rect_patch)
        
        # Connect
        if not is_custom[i]:
            con1 = ConnectionPatch(xyA=(y1, x1), coordsA=ax_main.transData, xyB=(0, 1), coordsB=ax_zoom.transAxes, color=color, lw=lw)
        else:
            con1 = ConnectionPatch(xyA=(y1, x2), coordsA=ax_main.transData, xyB=(0, 0), coordsB=ax_zoom.transAxes, color=color, lw=lw)
        
        con2 = ConnectionPatch(xyA=(y2, x1), coordsA=ax_main.transData, xyB=(1, 1), coordsB=ax_zoom.transAxes, color=color, lw=lw)
        fig.add_artist(con1)
        fig.add_artist(con2)

    plt.tight_layout()
    save_movie_frame(rel_ts_idx // skip, dpi=200)
    plt.close('all') # Clear memory within worker
    return ts_val

if __name__ == '__main__':
    start_time = time.time()
    
    # Parallel processing
    # Using 'bigmem' node, we can scale cores. 
    # Use os.cpu_count() or a fixed number based on memory constraints.
    num_cores = min(cpu_count(), 32) 
    task_indices = range(0, len(steps), skip)
    
    print(f"Starting parallel job with {num_cores} cores.")
    with Pool(processes=num_cores) as pool:
        pool.map(render_frame, task_indices)

    elapsed = time.time() - start_time
    print(f"Total computation time: {elapsed / 60.0:.2f} minutes")