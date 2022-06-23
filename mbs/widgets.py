from ipywidgets import interact, interactive_output, fixed, IntSlider, Checkbox, IntRangeSlider, FloatRangeSlider, Dropdown
import ipywidgets as widgets
from IPython.display import display
from ipykernel.pylab.backend_inline import flush_figures

from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

backends = Enum('Backends', 'ipympl inline')

def backend():
    back_d = {'module://ipympl.backend_nbagg': backends.ipympl,
              'module://ipykernel.pylab.backend_inline': backends.inline,
              'module://matplotlib_inline.backend_inline': backends.inline,
              }
    return back_d[plt.get_backend()]


class IsoenergyWidget(object):
    def __init__(specmap):
        pass


class EDC(object):
    pass


def specwidget(spec, ax=None, fig=None, **plot_kwargs):
    be = backend()
    output = widgets.Output()

    if not ax:
        with plt.ioff():
            #todo give specwidget id and remember to close
            fig, ax = plt.subplots(dpi=90, constrained_layout=True)
        with output:
            if be == backends.ipympl:
                #fig.canvas.resizable = True
                fig.canvas.toolbar_position = 'bottom'
                fig.canvas.header_visible = False
                display(fig.canvas)
            elif be == backends.inline:
                flush_figures()

    im = spec.plot(ax, **plot_kwargs)
    title = ax.set_title('')

    cmapcontrol = Dropdown(value='turbo', options=plt.colormaps(), description='Colormap')

    controls = widgets.VBox([
        widgets.HTML("<h1>hello</h1>"), #doesnt work
        widgets.HBox([cmapcontrol]),
        #    widgets.HBox([]),
        ])
    controls = cmapcontrol

    def update_spec(cmap):
        im.set_cmap(cmap)
        if be is backends.inline:
            output.clear_output(wait=True)
            with output:
                display(fig)

    cb_output = interactive_output(update_spec, {'cmap': cmapcontrol})

    plot_tab = widgets.VBox([controls, output])
    other_tab = widgets.Text('something')
    tabs = widgets.Tab(children=[plot_tab, other_tab])
    tabs.set_title(0, 'Spectrum')
    tabs.set_title(1, 'Other, EDC?')
    #metadata = widgets.GridBox([widgets.Label(x) for x in ['key', 'value']])

    #md_rows = "".join(["<tr>" + f"<td><b>{k}</b></td><td>{v}</td>" + "</tr>"
    #                   for k, v in spec.metadata.items()])
    #metadata = widgets.HTML(f"<table>{md_rows}</table>")

    #metadata = widgets.VBox(md_elements, layout=widgets.Layout(flex_flow='row wrap'))

    vbox_list = [widgets.HTML(f'<h4>{spec.name}</h4>'), tabs]

    extra_info = getattr(spec, '_extra_info_widgets')
    if extra_info is not None:
        extra_widgets, extra_titles = zip(*extra_info)
        #acc = widgets.Accordion(children=[metadata, widgets.Label('count rate, _view etc.')],
        #                        selected_index=None)
        acc = widgets.Accordion(children=extra_widgets, selected_index=None)
        for i, t in enumerate(extra_titles):
            acc.set_title(i, t)
        vbox_list.append(acc)

    vbox_list.append(cb_output)
    full_widget = widgets.VBox(vbox_list)

    #accordion = widgets.Accordion(children=[widgets.IntSlider(), widgets.Text()], titles=('Slider', 'Text'))
    return full_widget

def isowidget(specmap, ax=None, fl_default=None, width_default=None, dr_default=True, continuous_update=False, pmin=5, pmax=99.8, **plot_kwargs):
    be = backend()
    output = widgets.Output()

    if not ax:
        with plt.ioff():
            fig, ax = plt.subplots(dpi=90, constrained_layout=True)
        with output:
            if be == backends.ipympl:
                #fig.canvas.resizable = True
                fig.canvas.toolbar_position = 'bottom'
                fig.canvas.header_visible = False
                display(fig.canvas)
            elif be == backends.inline:
                flush_figures()

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
        title.set_text(f"E={s.energy_scale[fl]:.2f}±{width*s['Step Size']:.2f}eV\n{s.duration}")
        #flush_figures()

        if be is backends.inline:
            output.clear_output(wait=True)
            with output:
                display(fig)

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
    cb_output = interactive_output(update_fermimap, {'fl': flcontrol, 'width': widthcontrol, 'dr': drcontrol, 'clim_p': climcontrol})

    vbox_list = [controls, output]
    vbox_list.append(cb_output)
    full_widget = widgets.VBox(vbox_list)

    return full_widget

