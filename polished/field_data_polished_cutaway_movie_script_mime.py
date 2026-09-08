# nohup python field_data_polished_cutaway_movie_script_mime.py > cms.out 2>&1 &
# nohup python -u field_data_polished_cutaway_movie_script_mime.py 2>&1 | tee cms.out &
# nohup python -u field_data_polished_cutaway_movie_script_mime.py > cms.out 2>&1 &
# Input Settings

findex = -7                      # file chosen

skip = 4

plot_choice = "density" # "hall"
lazy = 2
contour_cheat = 2
counter = 0
buffer = 10
ybuffer = 450
lw = 2.3

var_to_plot = {'density': 'ndens', 'hall': 'ex'}[plot_choice]
cmap = {"density": "viridis", "hall": "seismic"}[plot_choice]
positive = {"density": True, "hall": False}[plot_choice]
vabsmax = {"density": 5.0, "hall": 0.1}[plot_choice]
ccolor = {"density": "white", "hall": "black"}[plot_choice]

split = 1 # 1 whole domain, 2 half domain

buffer = 10                     # x buffer in skin depths
steps = range(0, 221, 1)     # steps to include from 100 to 212


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

# Read Dynamic Movement Data

cs_file = f"{run.path}/crossings_data.npy"
cs_shift = np.load(cs_file)
cs_shift = np.concatenate((cs_shift, np.array([cs_shift[-1]]))) #hack, should not make a big difference

xpt_shift_loc = f"{run.path}/xpt_data.npy"
xpt_shift = np.load(xpt_shift_loc)

# Loop to generate movie

counter = 0

flh1 = None
flh2 = None
flh3 = None

for rel_ts in range(0, len(steps), skip):
    print(f"Started processing timestep {steps[rel_ts]} ({rel_ts} of {len(steps)})...")

    plt.close('all')

    fig = plt.figure(figsize=(36, 12))
    gs = GridSpec(2, 2, width_ratios=[1.5, 3], height_ratios=[1, 1], wspace=0.3, hspace=0.3)    

    title = f"$\Omega_i t = $ {round(rel_ts * run.gyrofreq / run.mime):05d}"
    ez_rep = np.nan

    def get_crec(width, height, center = [run.xlim / 2, run.ylim / 2]):
        x1 = center[0] - width / 2
        x2 = center[0] + width / 2
        y1 = center[1] - height / 2
        y2 = center[1] + height / 2
        return [x1, x2, y1, y2]


    #rects = [[run.xlim / 2 - 200, run.xlim / 2 + 200, 1300, 1900], get_crec(240, 480)]

    left_x_center = cs_shift[steps[rel_ts]]
    right_x_center = cs_shift[steps[rel_ts]]
    right_y_center = xpt_shift[steps[rel_ts]]
    

    rects = [[left_x_center - 200, left_x_center + 200, 1200 - 300, 2000 - 300],
              [right_x_center - 120, right_x_center + 120, right_y_center - 240, right_y_center + 240]]
    colors = ['yellow', 'red']
    titles = ['Zoom 2', 'Zoom 1']

    def plot_main(ax):
        global flh1
        global ez_rep
        
        #roi = fq_cut(d.fields, rel_ts, [0 + buffer, run.xlim - buffer, 0, run.ylim / 2])
        roi = fq_cut(d.fields, rel_ts, [0 + buffer, run.xlim - buffer, 0 + ybuffer, run.ylim - ybuffer])
        ez_rep = roi.ez.mean() * skip
        #ez_rep = roi.ez[0, 0]
        print(f"ez_rep: {ez_rep.compute().item()}")

        im = plot_fq3_flipped(roi, roi[var_to_plot], plot = ax, vabsmax = vabsmax, positive = positive, downsample = lazy, cmap = cmap)

        if (flh1 is None):
            flh1 = FieldLineHelper(run, 5, contour_ds = contour_cheat)
        flh1.plot_field_lines(roi, ez_rep, plot = ax, arrow_spacing = -1, linewidth = 1, transpose = True, color = ccolor) #1000

        ax.set_ylim(run.xlim - buffer, buffer)
        ax.set_title(title)
        #ax.set_title("Main Image")
        return im

    def plot_zoom(ax, rect, color, title, flh_index):
        global flh2, flh3
        curr_flh = [flh2, flh3][flh_index]

        [y1, y2, x1, x2] = rect
        roi = fq_cut(d.fields, rel_ts, rect)
        plot_fq3_flipped(roi, roi[var_to_plot], plot = ax, vabsmax = vabsmax, positive = positive, downsample = lazy, cmap = cmap)

        if (curr_flh is None):
            if flh_index == 0:
                flh2 = FieldLineHelper(run, 4, contour_ds = contour_cheat)
                curr_flh = flh2
            else:
                flh3 = FieldLineHelper(run, 4, contour_ds = contour_cheat)  
                curr_flh = flh3

        curr_flh.plot_field_lines(roi, ez_rep, plot = ax, arrow_spacing = -1, transpose = True, color = ccolor)

        #ax.set_xlim(y1, y2)
        #ax.set_ylim(x1, x2)
        #ax.set_title(title)
        # Show axis ticks and labels
        ax.tick_params(axis='both', which='both', direction='in', length=4)
        ax.set_xlabel("$y$ [$c/\\omega_{p}$]")
        ax.set_ylabel("$x$ [$c/\\omega_{p}$]")
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=lw)
        return rect_patch

    def connect_rect(fig, ax_main, ax_zoom, rect, color, custom = False):
        [x1, x2, y1, y2] = rect

        if not custom:
            con1 = ConnectionPatch(xyA=(y1, x1), coordsA=ax_main.transData,
                                xyB=(0, 1), coordsB=ax_zoom.transAxes,
                                color=color, lw = lw)
        else:
            con1 = ConnectionPatch(xyA=(y1, x2), coordsA=ax_main.transData,
                                xyB=(0, 0), coordsB=ax_zoom.transAxes,
                                color=color, lw = lw)
        
        con2 = ConnectionPatch(xyA=(y2, x1), coordsA=ax_main.transData,
                                xyB=(1, 1), coordsB=ax_zoom.transAxes,
                                color=color, lw = lw)


        fig.add_artist(con1)
        fig.add_artist(con2)


    ax_main = fig.add_subplot(gs[0, :])
    im = plot_main(ax_main)
    is_custom = [False, True]

    zoom_axes = []
    for i, (rect, color, title) in enumerate(zip(rects, colors, titles)):
        ax_zoom = fig.add_subplot(gs[1, i])
        rect_patch = plot_zoom(ax_zoom, rect, color, title, i)
        ax_main.add_patch(rect_patch)
        connect_rect(fig, ax_main, ax_zoom, rect, color, custom = is_custom[i])
        zoom_axes.append(ax_zoom)

    plt.tight_layout() #maybe disable

    save_movie_frame(counter, dpi = 200)

    
    plt.show()
    #plt.close(fig)

    print(f"Finished processing timestep {steps[rel_ts]} ({rel_ts} of {len(steps)}).")

    counter += 1