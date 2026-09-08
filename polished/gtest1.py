# nohup python field_data_polished_cutaway_movie_script_optimized.py > cms.out 2>&1 &
# nohup python field_data_polished_cutaway_movie_script_optimized.py 2>&1 | tee cms.out &

# Input Settings
findex = 140                      
large_field_downsampling = 2      
large_contour_downsampling = 4    
large_num_contours = 8            

small_field_downsampling = 1      
small_contour_downsampling = 1    
left_num_contours = 8             
right_num_contours = 3

cmap = 'inferno'
dens_cap = 3.5
split = 1 
buffer = 10                     
steps = range(100, 212, 1)     

# Imports and Folder Selection
from graphet import Data
from graphet.plugins import TristanV2 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.append('../v2_proc_files')
from useful_funcs import *
from TristanRun import TristanRun
from FieldLineHelper import FieldLineHelper

from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

import os
import time
import atexit

start_time = time.time()
plt.switch_backend('Agg')

def print_total_time():
    elapsed = time.time() - start_time
    print(f"Total computation time: {elapsed / 60.0:.2f} minutes")

atexit.register(print_total_time)
set_standard_plot_params()

# Read Input File Data
run = TristanRun(findex)
d = run.config_data(steps, "fields", includes_pressure = False)

# --- INITIALIZE STATIC PLOT ELEMENTS ---
fig = plt.figure(figsize=(36, 12))
gs = GridSpec(2, 2, width_ratios=[1.5, 3], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
ax_main = fig.add_subplot(gs[0, :])
ax_zoom1 = fig.add_subplot(gs[1, 0])
ax_zoom2 = fig.add_subplot(gs[1, 1])
zoom_axes = [ax_zoom1, ax_zoom2]

def get_crec(width, height, center = [run.xlim / 2, run.ylim / 2]):
    return [center[0] - width / 2, center[0] + width / 2, center[1] - height / 2, center[1] + height / 2]

rects = [[run.xlim / 2 - 200, run.xlim / 2 + 200, 1300, 1900], get_crec(240, 480)]
colors = ['yellow', 'red']
titles = ['Zoom 2', 'Zoom 1']

# Initialize Helpers once
flh1 = FieldLineHelper(run, large_num_contours, contour_ds = large_contour_downsampling)
flh2 = FieldLineHelper(run, left_num_contours, contour_ds = small_contour_downsampling)
flh3 = FieldLineHelper(run, right_num_contours, contour_ds = small_contour_downsampling)
flhs = [flh2, flh3]

counter = 0

# Loop to generate movie
for rel_ts in range(len(steps)):
    # Clear axes for fresh data but keep the figure/axes objects
    ax_main.cla()
    for ax in zoom_axes:
        ax.cla()

    title = f"$\Omega t = $ {round(rel_ts * run.gyrofreq):05d}"
    
    # --- MAIN PLOT ---
    roi_main = fq_cut(d.fields, rel_ts, [0 + buffer, (run.xlim - buffer) / split, 0 + buffer, (run.ylim - buffer) / split])
    ez_rep = roi_main.ez.mean()

    plot_fq3_flipped(roi_main, roi_main.dens, plot = ax_main, vabsmax = dens_cap, 
                     positive = True, downsample = large_field_downsampling, cmap = cmap)
    flh1.plot_field_lines(roi_main, ez_rep, plot = ax_main, arrow_spacing = -1, linewidth = 0.9, transpose = True)

    ax_main.set_ylim(buffer, run.xlim - buffer)
    ax_main.set_title(title)

    # --- ZOOM PLOTS ---
    for i, (ax_z, rect, color, flh_z) in enumerate(zip(zoom_axes, rects, colors, flhs)):
        [y1, y2, x1, x2] = rect
        roi_z = fq_cut(d.fields, rel_ts, rect)
        
        plot_fq3_flipped(roi_z, roi_z.dens, plot = ax_z, vabsmax = dens_cap, 
                         positive = True, downsample = small_field_downsampling, cmap = cmap)
        flh_z.plot_field_lines(roi_z, ez_rep, plot = ax_z, arrow_spacing = -1, transpose = True, linewidth = 0.75)

        # Formatting
        ax_z.tick_params(axis='both', which='both', direction='in', length=4)
        ax_z.set_xlabel("$y$ [$c/\\omega_{p}$]")
        ax_z.set_ylabel("$x$ [$c/\\omega_{p}$]")
        
        # Add Rectangle to main plot
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=1.5)
        ax_main.add_patch(rect_patch)
        
        # Connections
        con1 = ConnectionPatch(xyA=(y1, x1), coordsA=ax_main.transData,
                            xyB=(0, 1), coordsB=ax_z.transAxes, color=color)
        con2 = ConnectionPatch(xyA=(y2, x1), coordsA=ax_main.transData,
                            xyB=(1, 1), coordsB=ax_z.transAxes, color=color)
        fig.add_artist(con1)
        fig.add_artist(con2)

    # Save and output progress
    save_movie_frame(counter, dpi = 200)
    
    # Clean up figure-level artists (connections) so they don't stack up and slow down the loop
    fig.artists.clear()

    print(f"Frame {counter} complete")
    counter += 1

plt.close(fig)