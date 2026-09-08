# nohup python field_data_polished_cutaway_movie_script.py > cms.out 2>&1 &
# nohup python field_data_polished_cutaway_movie_script.py 2>&1 | tee cms.out &
# Input Settings

findex = 140                      # file chosen

large_field_downsampling = 2      #2
large_contour_downsampling = 4    # 4
large_num_contours = 8            # 8

small_field_downsampling = 1      #1
small_contour_downsampling = 1    #1
left_num_contours = 8             #8
right_num_contours = 3

cmap = 'inferno'
dens_cap = 3.5

split = 1 # 1 whole domain, 2 half domain

buffer = 10                     # x buffer in skin depths
steps = range(100, 212, 1)     # steps to include from 100 to 212


# Imports and Folder Selection

from graphet import Data
from graphet.plugins import TristanV2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.append('..')
from useful_funcs import *
from fmgmt.TristanRun import TristanRun
from fmgmt.FieldLineHelper import FieldLineHelper

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

# Loop to generate movie

counter = 0

flh1 = None
flh2 = None
flh3 = None

for rel_ts in range(len(steps)):
    fig = plt.figure(figsize=(36, 12))
    gs = GridSpec(2, 2, width_ratios=[1.5, 3], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    title = f"$\Omega t = $ {round(rel_ts * run.gyrofreq):05d}"
    ez_rep = np.nan

    def get_crec(width, height, center = [run.xlim / 2, run.ylim / 2]):
        x1 = center[0] - width / 2
        x2 = center[0] + width / 2
        y1 = center[1] - height / 2
        y2 = center[1] + height / 2
        return [x1, x2, y1, y2]


    rects = [[run.xlim / 2 - 200, run.xlim / 2 + 200, 1300, 1900], get_crec(240, 480)]
    colors = ['yellow', 'red']
    titles = ['Zoom 2', 'Zoom 1']

    def plot_main(ax):
        global flh1, ez_rep
        roi = fq_cut(d.fields, rel_ts, [0 + buffer, (run.xlim - buffer) / split, 0 + buffer, (run.ylim - buffer) / split])
        ez_rep = roi.ez.mean()

        im = plot_fq3_flipped(roi, roi.dens, plot = ax, vabsmax = dens_cap, positive = True, downsample = large_field_downsampling, cmap = cmap)
        if (flh1 is None):
            flh1 = FieldLineHelper(run, large_num_contours, contour_ds = large_contour_downsampling)
        flh1.plot_field_lines(roi, ez_rep, plot = ax, arrow_spacing = -1, linewidth = 0.9, transpose = True) #1000

        ax.set_ylim(buffer, run.xlim - buffer)
        ax.set_title(title)
        #ax.set_title("Main Image")
        return im

    def plot_zoom(ax, rect, color, title, flh_index):
        global flh2, flh3
        curr_flh = [flh2, flh3][flh_index]

        [y1, y2, x1, x2] = rect
        roi = fq_cut(d.fields, rel_ts, rect)
        plot_fq3_flipped(roi, roi.dens, plot = ax, vabsmax = dens_cap, positive = True, downsample = small_field_downsampling, cmap = cmap)

        if (curr_flh is None):
            if flh_index == 0:
                flh2 = FieldLineHelper(run, left_num_contours, contour_ds = small_contour_downsampling)
                curr_flh = flh2
            else:
                flh3 = FieldLineHelper(run, right_num_contours, contour_ds = small_contour_downsampling)
                curr_flh = flh3
        
        curr_flh.plot_field_lines(roi, ez_rep, plot = ax, arrow_spacing = -1, transpose = True, linewidth = 0.75)

        #ax.set_xlim(y1, y2)
        #ax.set_ylim(x1, x2)
        #ax.set_title(title)
        # Show axis ticks and labels
        ax.tick_params(axis='both', which='both', direction='in', length=4)
        ax.set_xlabel("$y$ [$c/\\omega_{p}$]")
        ax.set_ylabel("$x$ [$c/\\omega_{p}$]")
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=1.5)
        return rect_patch

    def connect_rect(fig, ax_main, ax_zoom, rect, color):
        [x1, x2, y1, y2] = rect
        con1 = ConnectionPatch(xyA=(y1, x1), coordsA=ax_main.transData,
                            xyB=(0, 1), coordsB=ax_zoom.transAxes,
                            color=color)
        con2 = ConnectionPatch(xyA=(y2, x1), coordsA=ax_main.transData,
                            xyB=(1, 1), coordsB=ax_zoom.transAxes,
                            color=color)
        fig.add_artist(con1)
        fig.add_artist(con2)

    #plt.suptitle(title)

    ax_main = fig.add_subplot(gs[0, :])
    im = plot_main(ax_main)

    zoom_axes = []
    for i, (rect, color, title) in enumerate(zip(rects, colors, titles)):
        ax_zoom = fig.add_subplot(gs[1, i])
        rect_patch = plot_zoom(ax_zoom, rect, color, title, i)
        ax_main.add_patch(rect_patch)
        connect_rect(fig, ax_main, ax_zoom, rect, color)
        zoom_axes.append(ax_zoom)

    #save_polished_fig(name = "bigrun_zooms", dpi = 400)
    save_movie_frame(counter, dpi = 200)

    plt.tight_layout()
    plt.close(fig)

    print(f"Frame {counter} complete")

    counter += 1