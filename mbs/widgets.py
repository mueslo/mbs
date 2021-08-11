from ipywidgets import interact, interactive_output, fixed, IntSlider, Checkbox, IntRangeSlider, FloatRangeSlider
import ipywidgets as widgets
from IPython.display import display
from ipykernel.pylab.backend_inline import flush_figures

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

from mbs.load import stats

class IsoenergyWidget(object):
    def __init__(specmap):
        pass

def isowidget(specmap, ax=None, fl_default=None, width_default=None, dr_default=True, continuous_update=False, pmin=5, pmax=99.8, **plot_kwargs):
    #assert plt.get_backend() == 'module://ipympl.backend_nbagg'
    output = widgets.Output()
    with output:
        if not ax:
            fig, ax = plt.subplots(dpi=90, constrained_layout=True)
            #fig.canvas.resizable = True
            fig.canvas.toolbar_position = 'bottom'
            fig.canvas.header_visible = False

    s = specmap.spectra[0]
    plot_kwargs.setdefault('dither_repair', dr_default)
    fp = specmap.plot(ax, fl=len(s.data)//2, width=1, **plot_kwargs)
    title = ax.set_title('')

    fl_default = fl_default or len(s.data)/2
    width_default = width_default or 10
    flcontrol = IntSlider(description='Fermi level', value=fl_default, min=0, max=len(s.data)-1, step=1, continuous_update=continuous_update)
    widthcontrol = IntSlider(description='Window', value=width_default, min=1, max=len(s.data), step=1, continuous_update=continuous_update)
    drcontrol = Checkbox(description='DR', value=dr_default)
    climcontrol = FloatRangeSlider(description='CL (\%ile)', value=[pmin, pmax], min=0., max=100., step=0.1, continuous_update=continuous_update)
    controls = widgets.VBox([
        widgets.HBox([flcontrol, climcontrol]),
        widgets.HBox([widthcontrol, drcontrol, ])])

    #def fmap_set_clim(pmin, pmax):
    #    fp.set_clim(np.percentile(fp._A, pmin), np.percentile(fp._A, pmax))

    #blur=np.array([0.25, 0.25]) #deg
    #blur=blur/np.array([specmap.angles[1]-specmap.angles[0], s._xscale.step]) #deg to i

    def update_fermimap(fl, width, dr, clim_p):
        fmap = specmap.generate_fermimap(fl=fl, width=width, dither_repair=dr)
        #fmap_blur = gaussian_filter(fmap, blur)
        fp.set_data(fmap)
        #print(fl, width)
        #fp.set_clim(np.percentile(fmap, pmin), np.percentile(fmap, pmax))
        #fmap_set_clim(pmin, pmax)
        pmin, pmax = clim_p
        fp.set_clim(np.percentile(fmap, pmin), np.percentile(fmap, pmax))
        title.set_text(f"E={s.energy_scale[fl]:.2f}±{width*s['Step Size']:.2f}eV\n{stats(s)['duration']}")
        flush_figures()

    #rangecontrol = IntRangeSlider(
    #    value=[fl_default-width_default, fl_default+width_default],
    #    min=0,
    #    max=len(s.data)-1,
    #    step=1,
    #    description='Range:',
    #    continuous_update=continuous_update,
    #    orientation='horizontal',
    #    state='disabled', 
    #    #readout=True,
    #    #readout_format='d',
    #)

    #def on_value_change(change):
    #    rangecontrol.value = [flcontrol.value-widthcontrol.value, flcontrol.value+widthcontrol.value]
    #def on_value_change2(change):
    #    print(change['owner'])
    #    widthcontrol.value, flcontrol.value = (rangecontrol.value[1] - rangecontrol.value[0]) // 2, (rangecontrol.value[1] + rangecontrol.value[0]) // 2

    #flcontrol.observe(on_value_change, names='value')
    #widthcontrol.observe(on_value_change, names='value')
    #rangecontrol.observe(on_value_change2, names='value')
    interactive_output(update_fermimap, {'fl': flcontrol, 'width': widthcontrol, 'dr': drcontrol, 'clim_p': climcontrol})

    return widgets.VBox([controls, output])

