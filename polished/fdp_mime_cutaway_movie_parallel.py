# To run serially: nohup python -u field_data_polished_cutaway_movie_script_mime.py > cms.out 2>&1 &
# To start interactive node: salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --time=1:00:00 --mem=3000G --partition=bigmem -A pppl

## INPUT SETTINGS

findex = -7                      # file chosen

skip = 2

NUM_WORKERS = 32

plot_choice = "density" # "hall"
lazy = 2
contour_cheat = 2
counter = 0
buffer = 10
ybuffer = 450
lw = 2.3

main_contours = 5
lzoom_contours = 4
rzoom_contours = 4

var_to_plot = {'density': 'ndens', 'hall': 'ex'}[plot_choice]
cmap = {"density": "viridis", "hall": "seismic"}[plot_choice]
positive = {"density": True, "hall": False}[plot_choice]
vabsmax = {"density": 5.0, "hall": 0.1}[plot_choice]
ccolor = {"density": "white", "hall": "black"}[plot_choice]

split = 1 # 1 whole domain, 2 half domain

buffer = 10                     # x buffer in skin depths
steps = range(0, 221, 1)     # steps to include from 100 to 212


# Imports and Folder Selection

import multiprocessing
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
from useful_funcs import set_standard_plot_params, fq_cut, plot_fq3_flipped, save_movie_frame
from TristanRun import TristanRun
from FieldLineHelperParallel import FieldLineHelperParallel

from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

from scipy.integrate import cumulative_trapezoid

import concurrent.futures

import os
import time
import atexit

# Read Input File Data

run = TristanRun(findex)

plt.rcParams['font.family'] = 'serif'
fig_dummy = plt.figure()
plt.close(fig_dummy)

plt.switch_backend('Agg')

def print_total_time():
    elapsed = time.time() - start_time
    print(f"Total computation time: {elapsed / 60.0:.2f} minutes")

def compute_gauge_integrals(ez_reps):
    return run.cc * run.output_interval * cumulative_trapezoid(ez_reps, initial=0)

def plot_fq3_flipped(roi, quantity, plot, vabsmax=-1, cmap='viridis', downsample=1, positive=False, cbar_label=None):

    # Downsample the data and transpose
    if downsample > 1:
        quantity = quantity[::downsample, ::downsample]  # Transpose
        roi_x = roi.x[::downsample]  # Flip axes
        roi_y = roi.y[::downsample]
    else:
        quantity = quantity  # Transpose
        roi_x = roi.x
        roi_y = roi.y

    xmin, xmax, ymin, ymax = min(roi_x), max(roi_x), min(roi_y), max(roi_y)

    flipped_extent = [ymin, ymax, xmax, xmin]

    if positive:
        mesh = plot.imshow(quantity.T, cmap=cmap, vmax=vabsmax, vmin=0, origin='upper', extent=flipped_extent)
    elif vabsmax == -1:
        mesh = plot.imshow(quantity.T, cmap=cmap, origin='upper', extent=flipped_extent)
    else:
        mesh = plot.imshow(quantity.T, cmap=cmap, vmax=vabsmax, vmin=-vabsmax, origin='upper', extent=flipped_extent)

    plot.set_xlabel('$y$ [$c/\omega_{p}$]')
    plot.set_ylabel('$x$ [$c/\omega_{p}$]')
    
    return mesh

def save_nth_frame(frame_num, rel_ts, *,  gauge_integrals, left_x_center, 
                   right_x_center = run.xlim / 2, 
                   right_y_center = run.ylim / 2, flhs, DEBUG = False):

    [gauge_integral_main, gauge_integral_left, gauge_integral_right] = gauge_integrals

    d = run.config_data([steps[rel_ts]], "fields", includes_pressure = False)

    # i is the frmae #, rel ts is what goes into fq_cut
    if DEBUG:
        print(f"Processing frame {frame_num}: rel_ts = {rel_ts}, gauge_integral = {gauge_integral_main}...")

    [flh1, flh2, flh3] = flhs

    fig = plt.figure(figsize=(36, 12))
    gs = GridSpec(2, 2, width_ratios=[1.5, 3], height_ratios=[1, 1], wspace=0.3, hspace=0.3)    

    title = f"$\Omega_i t = $ {round(rel_ts * run.gyrofreq / run.mime):05d}"

    main_rect = [0 + buffer, run.xlim - buffer, 0 + ybuffer, run.ylim - ybuffer]
    left_rect = [left_x_center - 200, left_x_center + 200, 1200 - 300, 2000 - 300]
    right_rect = [right_x_center - 120, right_x_center + 120, right_y_center - 240, right_y_center + 240]

    rects = [left_rect, right_rect]

    colors = ['yellow', 'red']
    titles = ['Zoom 2', 'Zoom 1']

    def plot_main(ax, flh1):
        print(f"Worker {os.getpid()}: Slicing data...", flush=True)
        roi = fq_cut(d.fields, 0, main_rect)
        

        im = plot_fq3_flipped(roi, roi[var_to_plot], plot = ax, vabsmax = vabsmax, positive = positive, downsample = lazy, cmap = cmap) #parallel code gets stuck here, not sure how to fix
        #ax.text(0.5, 0.5, "Testing")
        im = None

        if (flh1 is None):
            flh1 = FieldLineHelperParallel(run, main_contours, contour_ds = contour_cheat)
        flh1.plot_field_lines(roi, gauge_integral_main, plot = ax, arrow_spacing = -1, linewidth = 1, transpose = True, color = ccolor) #1000

        print(f"Worker {os.getpid()}: Data sliced. Plotting...", flush=True)

        ax.set_ylim(run.xlim - buffer, buffer)
        ax.set_title(title)
        return im, flh1

    def plot_zoom(ax, rect, color, title, flh_index, flh23):
        [y1, y2, x1, x2] = rect
        roi = fq_cut(d.fields, 0, rect)
        plot_fq3_flipped(roi, roi[var_to_plot], plot = ax, vabsmax = vabsmax, positive = positive, downsample = lazy, cmap = cmap)

        if (flh23 is None):
            if flh_index == 0:
                flh2 = FieldLineHelperParallel(run, lzoom_contours, contour_ds = contour_cheat)
                flh23 = flh2
            else:
                flh3 = FieldLineHelperParallel(run, rzoom_contours, contour_ds = contour_cheat)  
                flh23 = flh3

        gauge_integral = [gauge_integral_left, gauge_integral_right][flh_index]

        flh23.plot_field_lines(roi, gauge_integral, plot = ax, arrow_spacing = -1, transpose = True, color = ccolor)

        # Show axis ticks and labels
        ax.tick_params(axis='both', which='both', direction='in', length=4)
        ax.set_xlabel("$y$ [$c/\\omega_{p}$]")
        ax.set_ylabel("$x$ [$c/\\omega_{p}$]")
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=lw)
        return rect_patch, flh23

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



    ax_main = fig.add_subplot(gs[0, :]) # problem line
    im, flh1 = plot_main(ax_main, flh1)

    is_custom = [False, True]

    zoom_axes = []
    for i, (rect, color, title) in enumerate(zip(rects, colors, titles)):
        ax_zoom = fig.add_subplot(gs[1, i])
        rect_patch, flh23 = plot_zoom(ax_zoom, rect, color, title, i, [flh2, flh3][i])

        if i == 0:
            flh2 = flh23
        else:
            flh3 = flh23   

        ax_main.add_patch(rect_patch)
        connect_rect(fig, ax_main, ax_zoom, rect, color, custom = is_custom[i])
        zoom_axes.append(ax_zoom)

    #plt.tight_layout() #maybe disable


    save_movie_frame(frame_num, dpi = 200)
    
    plt.close(fig)
    return flh1, flh2, flh3

def render_frame_worker(i, ts, gauge_integral, flhs, lx, rx, ry):
    save_nth_frame(
        i, ts, 
        gauge_integral=gauge_integral, 
        flhs=flhs, 
        left_x_center=lx, 
        right_x_center=rx, 
        right_y_center=ry, 
        DEBUG=True
    )

def render_frame_worker(i, ts, gauge_integrals, flhs, lx, rx, ry):
    print(f"--- Worker started: frame {i} (ts {ts}) ---", flush=True)
    try:
        # Check if we are stuck BEFORE saving
        print(f"Worker {i}: Calling save_nth_frame...", flush=True)
        
        save_nth_frame(
            i, ts, 
            gauge_integrals=gauge_integrals, 
            flhs=flhs, 
            left_x_center=lx, 
            right_x_center=rx, 
            right_y_center=ry, 
            DEBUG=True
        )
        
        print(f"--- Worker finished: frame {i} ---", flush=True)
    except Exception as e:
        print(f"!!! Worker {i} CRASHED: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    start_time = time.time()
    atexit.register(print_total_time)

    #plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    ## READ GAUGE DATA

    print("Reading auxiliary data...", end = "")

    cs_file = f"{run.path}/crossings_data.npy"
    cs_shift = np.load(cs_file)
    cs_shift = np.concatenate((cs_shift, np.array([cs_shift[-1]]))) #hack, should not make a big difference

    xpt_shift_loc = f"{run.path}/xpt_data.npy"
    xpt_shift = np.load(xpt_shift_loc)


    # this is the tricky part, the representative electric field to evolve the field lines. currently I generate them in field_data_polished_cutaway_mass_ratio.ipynb
    ez_reps_main = np.load(f"{run.path}/ez_reps_main.npy")
    ez_reps_left = np.load(f"{run.path}/ez_reps_left.npy")
    ez_reps_right = np.load(f"{run.path}/ez_reps_main.npy") #hack but might help

    gauge_integrals_main = compute_gauge_integrals(ez_reps_main)
    gauge_integrals_left = compute_gauge_integrals(ez_reps_left)
    gauge_integrals_right = compute_gauge_integrals(ez_reps_right)

    if np.min([len(a) for a in [cs_shift, xpt_shift, gauge_integrals_main]]) < len(steps):
        raise ValueError(f"Not enough auxiliary data timesteps: cs_shift {len(cs_shift)}, xpt_shift {len(xpt_shift)}, gauge_integrals {len(gauge_integrals_main)}, steps {len(steps)}")

    print("done.")

    ## INITIALIZATION AND FIRST FRAME

    print("Initializing and rendering first frame...", end = "")

    timesteps_to_render = range(0, len(steps), skip) #steps are what is fed to get d.fields

    left_x_center = cs_shift[0]
    right_x_center = cs_shift[0]
    right_y_center = xpt_shift[0]
    gauge_integrals = [0.0, 0.0, 0.0]

    flh1, flh2, flh3 = save_nth_frame(0, 0, gauge_integrals = gauge_integrals, left_x_center = left_x_center,
                    right_x_center = right_x_center, right_y_center = right_y_center, flhs = [None, None, None], DEBUG = True)

    flhs = [flh1, flh2, flh3]

    print("done.")

    ## SUBSEQUENT FRAMES

    print("Rendering subsequent frames...")

    indices = range(1, len(timesteps_to_render))

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                render_frame_worker,
                i,
                timesteps_to_render[i],
                [gauge_integrals_main[timesteps_to_render[i]], gauge_integrals_left[timesteps_to_render[i]], gauge_integrals_right[timesteps_to_render[i]]],
                flhs,
                cs_shift[timesteps_to_render[i]],
                cs_shift[timesteps_to_render[i]],
                xpt_shift[timesteps_to_render[i]]
            )
            for i in indices
        ]
