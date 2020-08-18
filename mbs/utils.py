#!/usr/bin/env python

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from .load import parse_data, is_mbs_filename

cwd = os.getcwd()


def make_preview(ax, data, metadata, ax_edc=None, ax_mdc=None, **kwargs):
    energies, data = data[:, 0], data[:, 1:]
    xscale = [metadata['XScaleMin'], metadata['XScaleMax']], metadata['XScaleName']
    if metadata['YScaleMult'] == 0:
        yscale = [metadata['Start K.E.'], metadata['End K.E.']], 'Kinetic energy / eV'
    else:
        yscale = [metadata['YScaleMin'], metadata['YScaleMax']], metadata['YScaleName']

    extent = xscale[0] + yscale[0]
    
    if metadata['Lens Mode'] in ['L4Spat5', 'L4MSpat5']:
        summed_data = np.sum(data, axis=1)
        ax.plot(np.linspace(yscale[0][0], yscale[0][1], len(summed_data)), summed_data, lw=0.5)
        ax.set_xlabel(yscale[1])
        ax.set_ylabel('Counts')
        textcolor = 'k'

    else:
        kwargs.setdefault('vmin', np.percentile(data, 20))
        kwargs.setdefault('vmax', np.percentile(data, 95))
        kwargs.setdefault('cmap', 'inferno')
        ax.imshow(data, origin='lower', extent=extent, aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]), **kwargs)
        if ax_edc:
            summed_data = np.sum(data, axis=1)
            ax_edc.plot(summed_data,
                        np.linspace(yscale[0][0], yscale[0][1], len(summed_data)))
            ax_edc.margins(y=0)
            #ax_edc.axis('off')
        ax.set_xlabel(metadata['Lens Mode'] + ", " + xscale[1])
        ax.set_ylabel(yscale[1])
        textcolor = 'white'
    # plt.suptitle(metadata['Gen. Name'])
    # plt.title(metadata['Pass Energy'])
    info = "{}\n{} {}".format(
        metadata['Gen. Name'],
        metadata['Pass Energy'],
        metadata['AcqMode'])
    plt.text(.01, .99, info,
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes, color=textcolor, fontsize=8)

    
def plot_save_preview(fname, out_dir=None, **kwargs):
    zip_fname = kwargs.get('zip_fname')
    if fname.startswith(zip_fname):
        fname = fname[len(zip_fname)+1:]
    out_dir = out_dir or (os.path.dirname(fname) if zip_fname is None
                          else os.path.dirname(zip_fname))
    out_fname = os.path.join(out_dir, os.path.basename(fname+'.png'))
    
    if os.path.exists(out_fname):
        print(out_fname, 'already exists, skipping')
        return

    try:
        data, metadata = parse_data(fname, zip_fname=zip_fname)
    except Exception as e:
        print(fname, e)
        return

    if metadata['Lens Mode'] in ['L4Spat5', 'L4MSpat5']:
        f, ax = plt.subplots()
        make_preview(ax, data, metadata)
    else:
        #gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
        #gs.update() # set the spacing between axes.
        
        f, (ax, ax_edc) = plt.subplots(1, 2, 
                                       gridspec_kw={
                                           'width_ratios': [8, 1],
                                           'wspace': 0, 
                                           'hspace': 0},
                                      sharey=True)
        #ax_edc.get_yaxis().set_visible(False)
        make_preview(ax, data, metadata, ax_edc=ax_edc)
        
    #plt.tight_layout()
    print(out_fname)
    out_dir = os.path.dirname(out_fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_fname)
    plt.close()

def daemon(path, out_dir):
    import time
    import logging
    
    from watchdog.observers import Observer
    from watchdog.events import (LoggingEventHandler
, FileCreatedEvent, 
                                 FileModifiedEvent, FileMovedEvent)
    
    class PreviewEventHandler(LoggingEventHandler):
        def dispatch(self, event):
            fname = os.path.basename(event.src_path)
            if is_mbs_filename(event.src_path):
                super(LoggingEventHandler, self).dispatch(event)
            
        def on_any_event(self, event):
            if (isinstance(event, FileCreatedEvent) or 
                isinstance(event, FileModifiedEvent)):
                try:
                    plot_save_preview(event.src_path, out_dir=out_dir)
                except Exception as e:
                    print('encountered', e)
            else:
                print('ignoring', event)
    
    event_handler = PreviewEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print('Preview generator started.')
    print('Watching', path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping...')
        observer.stop()
    observer.join()
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='MBS measurement file or directory (default: current directory)', nargs='?', default=os.getcwd())
    parser.add_argument('-d', action='store_true', help='Daemon mode')
    parser.add_argument('out_dir', help='Directory to save generated previews', nargs='?', default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        raise Exception('path does not exist')
    elif os.path.isdir(args.path):
        print("Working directory:", cwd)
        print("Target directory:", args.path)
        
        if args.d:
            daemon(args.path, args.path)
        else:
            for fname in os.listdir(args.path):
                if not fname.endswith('.txt'):
                    continue
                plot_save_preview(os.path.join(args.path, fname), out_dir=args.out_dir or args.path)
            
    elif os.path.isfile(args.path):
        if args.path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(args.path) as zf:
                fnames = zf.namelist()
            for fname in fnames:
                if not fname.endswith('.txt'):
                    continue
                plot_save_preview(fname, zip_fname=args.path, out_dir=args.out_dir or os.path.join(os.path.dirname(args.path),
                                                                                                   os.path.dirname(fname)))
        else:
            plot_save_preview(args.path, out_dir=args.out_dir or os.path.dirname(args.path))
    else:
        raise Exception('wat')
