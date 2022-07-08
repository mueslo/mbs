import operator
from functools import reduce
from os.path import splitext, basename
from collections import namedtuple, OrderedDict
from enum import Enum
from copy import copy as shallow_copy
from numbers import Number

from .io import parse_data, parse_lines, parse_info, frame_unit
from .krx import KRXFile
from .utils import fl_guess
from . import corrections as correct

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.ndimage import gaussian_filter

scale = namedtuple('Scale', ['min', 'max', 'step', 'name'])
AcqMode = Enum('AcquisitionMode', 'Fixed Swept Dither')

class AbstractSpectrum(NDArrayOperatorsMixin):
    def __init__(self, data, axes, metadata=None):
        """
        axes = Tuple[axis]
        axis = name, range (, label)
        metadata = Dict[String] -> Object"""
        self.data = data
        self.axes = axes  # first axis should be energy
        self.metadata = metadata or {}
        self.compat_keys = set()  # keys that must be the same to perform arithmetic


    def __array__(self):
        return self.data

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        #print(method, ufunc, args, kwargs)
        inputs = []
        for arg in args:
            # In this case we accept only scalar numbers or DiagonalArrays.
            if isinstance(arg, Number):
                inputs.append(arg)
            elif isinstance(arg, np.ndarray):
                assert arg.shape[-1] == len(self.y)
                inputs.append(arg)
            elif isinstance(arg, self.__class__):
                assert np.allclose(self.lens_scale, arg.lens_scale)
                assert np.allclose(self.energy_scale, arg.energy_scale)
                inputs.append(arg.data)
            else:
                return NotImplemented
        if method == '__call__':
            x = args
            return self.__class__(x, ufunc(*inputs, **kwargs))
        else:
            return NotImplemented

class Spectrum(AbstractSpectrum):
    def __init__(self, data, metadata):
        self._data = data
        self._view = []  # iterative slices of original array
        self._metadata = metadata
        self._path = None

        if self.acq_mode == AcqMode.Dither:
            self._view.append(
                (slice(0, -self['DithSteps']),) + (slice(None),) * (data.ndim - 1))

# todo: __copy__, __deepcopy__

    @classmethod
    def from_filename(cls, fname, zip_fname=None, **kwargs):
        if fname.endswith('.txt'):
            return cls.from_txt(fname, zip_fname, **kwargs)
        elif fname.endswith('.krx'):
            return cls.from_krx(fname, **kwargs)
        raise ValueError('Please use explicit format loaders if filename suffix was changed')

    @classmethod
    def from_txt(cls, fname, zip_fname=None):
        spec = cls(*parse_data(fname, zip_fname=zip_fname))
        spec._path = fname, None, zip_fname
        return spec

    @classmethod
    def from_krx(cls, fname, page=0):
        kf = KRXFile(fname)
        
        if page is None:  # return all pages
            return [cls.from_krx(fname, page=p) for p in range(kf.num_pages)]

        metadata = parse_lines(
            kf.page_metadata(page).splitlines(),
            metadata_only=True)
        spec = cls(data=kf.page(page), metadata=metadata)
        spec._path = fname, page, None
        return spec

    #@classmethod
    #def from_upload_widget(cls):
    #    widgets.FileUpload(  # this is upload, not file chooser
    #        accept='.krx,.txt',
    #        multiple=True  # multiple will be added
    #    )

    @property
    def xarray(self):
        from xarray import DataArray
        l = 'x' if not self['Lens Mode'].startswith('L4Ang') else 'phi'
        da = DataArray(self.data, dims=('e', l),
                       coords={'e': self.energy_scale, l: self.lens_scale},
                       attrs=self._metadata)
        return da

    @property
    def acq_mode(self):
        return AcqMode[self['AcqMode']]

    def _apply_view(self, x, view_axis=None):
        if view_axis is None:
            view_axis = slice(None)
        for v in self._view:
            x = x[v[view_axis]]
        return x

    @property
    def data(self):
        return self._apply_view(self._data)

    @property
    def masked_data(self):
        return np.ma.masked_array(self.data, mask=getattr(self, 'mask'))

    @property
    def info(self):
        if not self._path:
            return None

        fname, page, zip_fname = self._path
        info_path = [f"{splitext(fname)[0]}.info"]

        if page is not None:
            info_path.insert(0, f"{splitext(fname)[0]}_{page}.info")

        for p in info_path:
            try:
                return parse_info(p, zip_fname)
            except IOError:
                print(f'Could not find info file {p}')
                continue
        return None

    @property
    def _xscale(self):
        try:
            return scale(self['XScaleMin'], self['XScaleMax'],
                         self['XScaleMult'], self['XScaleName'])
        except KeyError:
            return scale(self['ScaleMin'], self['ScaleMax'],
                         self['ScaleMult'], self['ScaleName'])

    @property
    def _lens_scale(self):
        return np.linspace(self._xscale.min, self._xscale.max, self['NoS'])

    @property
    def lens_scale(self):
        return self._apply_view(self._lens_scale, view_axis=1)

    @property
    def lens_extent(self):
        return tuple(self.lens_scale[[0, -1]])

    def l_to_i(self, l, view=True):
        """Return array index for given lens coordinate l"""
        if l is None:
            return None
        lens_scale = self.lens_scale if view else self._lens_scale
        return (np.abs(lens_scale - l)).argmin()

    @property
    def _escale(self):
        # todo: afaict, MBS write the lower boundary of energy bins
        #       currently we do not correct for this, but this will lead to
        #       energy shifts proportional to 0.5 * step size
        return scale(self["Start K.E."], self["End K.E."] - self['Step Size'],
                     self['Step Size'], 'Energy')

    @property
    def _energy_scale(self):
        return np.linspace(self._escale.min, self._escale.max, len(self._data))

    @property
    def energy_scale(self):
        return self._apply_view(self._energy_scale, view_axis=0)

    @property
    def energy_extent(self):
        return tuple(self.energy_scale[[0, -1]])

    def e_to_i(self, e, view=True):
        """Return array index i for given energy e"""
        if e is None:
            return None
        energy_scale = self.energy_scale if view else self._energy_scale
        return (np.abs(energy_scale - e)).argmin()

    @property
    def name(self):
        return self['Gen. Name']

    def _ipython_display_(self):
        from .widgets import specwidget
        display(specwidget(self))

    def __repr__(self) -> str:
        path, page, zip_fname = self._path
        fname = basename(path)
        pagestr = f"[{page}]" if page is not None else ""
        return f"<Spectrum({fname}{pagestr})>"

    @property
    def _extra_info_widgets(self):
        import ipywidgets as widgets
        widgets_l = []

        md_elements = [widgets.HTML(f"<b>{k}</b>: {v}") for k, v in self.metadata.items()]
        metadata = widgets.GridBox(md_elements,
            layout=widgets.Layout(grid_template_columns="repeat(3, auto)"))
        widgets_l.append((metadata, 'Metadata'))

        if self.info is not None:
            info_elements = [widgets.HTML(f"<b>{k}</b>: {v}{u if u is not None else ''}")
                             for k, (v, u) in self.info.items()]
            info = widgets.GridBox(info_elements,
                layout=widgets.Layout(grid_template_columns="repeat(3, auto)"))
            widgets_l.append((info, 'Endstation parameters'))

        return widgets_l

    def _translate_slice(self, slicetuple):
        slice_energy, slice_lens = slicetuple
        # todo: slice(None)
        return (slice(*list(map(self.e_to_i, [slice_energy.start, slice_energy.stop]))),
                slice(*list(map(self.l_to_i, [slice_lens.start, slice_lens.stop]))))

    @property
    def metadata(self):
        return self.get_metadata()

    def get_metadata(self, item=None):
        if item is None:
            return self._metadata
        return self._metadata[item]

    @property
    def duration(self):
        """Wall-time clock duration of measurement, completely wrong for multi-region scans"""
        return self['TIMESTAMP:'] - self['STim']

    @property
    def acqtime(self):
        """Nominal and effective (signal) acquisition time based on measurement parameters"""

        if self['AcqMode'] == 'Fixed':
            acqtime = self['ActScans'] * self['Frames Per Step'] * frame_unit
            eff_acqtime = acqtime
        elif self['AcqMode'] == 'Swept':
            acqtime = self['ActScans'] * self['TotSteps'] * self['Frames Per Step'] * frame_unit
            eff_acqtime = self['ActScans'] * self['No. Steps'] * self['Frames Per Step'] * frame_unit
        elif self['AcqMode'] == 'Dither':
            acqtime = self['ActScans'] * self['TotSteps'] * self['Frames Per Step'] * frame_unit
            eff_acqtime = acqtime * (self['No. Steps'] - self['TotSteps']) / self['No. Steps']

        return acqtime, eff_acqtime

    def index_slice(self, slicetuple):
        """Slice in terms of indices"""
        spec = shallow_copy(self)
        spec._view = self._view.copy()  # non-shallow copy
        spec._view.append(slicetuple)

    def _symmetrize(self, i, method='cut'):
        """Symmetrize spectrum with respect to some lens array index (of current view)"""

        # pad/cut x such that i is in the middle
        if method == 'cut':
            size = min(self.data.shape[1]-1 - i, i)
            sl = slice(None), slice(i-size, i+size+1)
            spec = shallow_copy(self)
            spec._view = self._view.copy()  # non-shallow copy
            spec._data = np.empty(self._data.shape)
            spec._data.fill(np.nan)
            valid_data = self.data[sl]
            spec._view.append(sl)
            spec_data = spec._apply_view(spec._data)
            spec_data[:] = valid_data + valid_data[:, ::-1]
            return spec
        raise Exception('not implemented')

    def symmetrize(self, lens_coordinate, method='cut'):
        """Symmetrize spectrum with respect to some lens coordinate"""
        i = self.l_to_i(lens_coordinate)
        return self._symmetrize(i, method)

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = (key, slice(None))
        if isinstance(key, tuple):
            spec = shallow_copy(self)
            spec._view = self._view.copy()  # non-shallow copy
            spec._view.append(self._translate_slice(key))
            return spec

        elif isinstance(key, str):
            return self.get_metadata(key)

    def __add__(self, other):
        #print('add', other)
        if isinstance(other, Number):
            if other == 0:
                return self
            else:
                return type(self)(self._data + other, self._metadata)

        # assert angle/energy extent is the same
        if isinstance(other, Spectrum):
            assert np.allclose(self.lens_scale, other.lens_scale)
            assert np.allclose(self.energy_scale, other.energy_scale)
            assert self['Lens Mode'] == other['Lens Mode']

        if not isinstance(self, SpectrumSum):
            m = [self._metadata]
        else:
            m = self._metadata

        if isinstance(other, SpectrumSum):
            om = other._metadata
        elif isinstance(other, Spectrum):
            om = [other._metadata]
        else:
            return NotImplemented
        return SpectrumSum(self._data + other._data, m + om)

    def __radd__(self, other):
        return self.__add__(other)

    def dead_pixel_correction(self):
        if self.acq_mode is AcqMode.Swept and self.get_metadata('PCntON?') == 1:
            data = correct.dp_pcnt_swept(self._data)
            spec = shallow_copy(self)
            spec._data = data
            return spec
        elif self.acq_mode is AcqMode.Dither and self.get_metadata('PCntON?') == 1:
            data = correct.dp_pcnt_dither(self._data, self.get_metadata('DithSteps'))
            spec = shallow_copy(self)
            spec._data = data
            return spec
        else:
            raise Exception('not implemented')

    def plot(self, ax, angle_correction=1., **kwargs):
        extent = self.lens_extent + self.energy_extent
        kwargs.setdefault('cmap', 'gist_yarg')
        kwargs.setdefault('vmin', np.percentile(self.data, 5))
        kwargs.setdefault('vmax', np.percentile(self.data, 99.5))
        kwargs.setdefault('aspect', 'auto')
        kwargs.setdefault('extent', extent)
        kwargs.setdefault('origin', 'lower')
        im = ax.imshow(self.data, **kwargs)
        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.set_xlabel(self._xscale.name)
        return im

    @property
    def edc(self):
        # todo: EDC object with _ipython_display_
        return np.sum(self.data, axis=1)
        # todo: return self.sum(dim=self.lens_axis_name, keep_attrs=True)

    # todo edc/mdc setter for normalization?
    # todo2 instead maybe EDC spectrum object/ndarray wrapper

    def plot_edc(self, ax, e_f=None, norm=None, **kwargs):
        show_counts = kwargs.pop('show_counts', False)
        annotations = kwargs.pop('annotations', {})
        if e_f is not None:
            x_scale = e_f - self.energy_scale
            xlabel = r'$E_\mathrm{bind}$ / eV'
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
        else:
            x_scale = self.energy_scale
            xlabel = r'$E_\mathrm{kin}$ / eV'
        y_data = np.sum(self.data, axis=1)
        if norm is None:
            pass
        elif norm == 'max':
            y_data = y_data / y_data.max()
        elif norm == 'maxmin':
            y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())
        elif norm == 'sum':
            y_data = y_data/y_data.sum()
        else:
            raise NotImplementedError

        for x, (text, akw) in annotations.items():
            akw.setdefault('ha', 'center')
            akw.setdefault('va', 'bottom')
            akw.setdefault('rotation', 90)
            # y = y_data[np.argmin(np.abs(x_scale - x))]
            y = np.max(y_data[np.abs(x_scale-x) < 5])
            offset = 0.05 * np.max(y_data)
            ax.text(x, y+offset, text, **akw)

        lines = ax.plot(x_scale, y_data, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Intensity')
        if not show_counts:
            ax.set_yticks([])
        else:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        return lines

    def plot_k(self, ax, angle_correction=1., k_origin=None, Ef=None, V0=0, **kwargs):
        if not self['Lens Mode'].startswith('L4Ang'):
            raise Exception('Lens mode is not angular.')

        X = self.lens_scale * angle_correction
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        # Y2 = np.sqrt((Y-4)*np.cos(np.radians(X))**2+V0)*0.512 + np.sin(np.radians(30))*Y*5.068*10**-4
        Y2 = Y
        X2 = np.sqrt(Y - 4) * 0.512 * np.sin(np.radians(X))  # + np.cos(np.radians(30))*Y*5.068*10**-4
        kwargs.setdefault('cmap', 'gist_yarg')
        kwargs.setdefault('vmin', np.percentile(self.data, 5))
        kwargs.setdefault('vmax', np.percentile(self.data, 99.5))
        kwargs.setdefault('shading', 'gouraud')
        kwargs.setdefault('rasterized', True)
        if k_origin:
            X2 = X2 - k_origin

        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.set_xlabel(r'$k_\parallel$ / $1/\AA$')

        if Ef:
            Y2 = Y2 - Ef
            ax.set_ylabel(r'$E-E_\mathrm{F}$ / eV')

        im = ax.pcolormesh(X2, Y2, self.data, **kwargs)
        return im

    def get_focus(self):
        assert not self['Lens Mode'].startswith('L4Ang')
        focus = np.sum(self.data, axis=0)
        mean = np.average(self.lens_scale, weights=focus)
        std = np.sqrt(np.average((self.lens_scale - mean)**2, weights=focus))
        skew = np.average((self.lens_scale - mean)**3, weights=focus) / (std**3)
        return mean, std, skew


class SpectrumSum(Spectrum):
    @classmethod
    def from_spectra(cls, *spectra):
        return reduce(operator.add, spectra)

    @classmethod
    def from_filenames(cls, *fnames, zip_fname=None):
        spectra = [Spectrum(*parse_data(fname, zip_fname=zip_fname))
                   for fname in fnames]
        return cls.from_spectra(*spectra)

    @property
    def md_keys(self):
        md_keys = self._metadata[0].keys()
        assert all(md.keys() == md_keys for md in self._metadata)
        return md_keys

    @property
    def metadata(self):
        return self.get_metadata(combine=True)

    # todo: extensive/intensive metadata
    # sum actscans if all other exposure settings are equal
    # (AddFMS

    def get_metadata(self, item=None, combine=True):
        if not item:
            return OrderedDict((k, self.get_metadata(k, combine))
                               for k in self.md_keys)

        vals = np.array([m[item] for m in self._metadata])
        no_combine = set(['ActScans', 'Gen. Name', 'STim', 'TIMESTAMP:'])
        combine = combine and item not in no_combine
        #if combine and (not vals or vals.count(vals[0]) == len(vals)):
        if combine and (vals[0] == vals).all():
            return vals[0]

        ignore = set(['No Scans'])
        if not item in ignore | no_combine:
            print(f'Warning: metadata values {item} differ for SpectrumSum summands')
        return vals

    @property
    def name(self):
        return 'Sum of ' + ', '.join(self['Gen. Name'])


class SpectrumMap(object):  # 1D parameter space only (for now)
    _param_name = 'params'
    def __init__(self, spectra, **kwargs):
        self._param_name = kwargs.get('param_name', self._param_name)
        params = kwargs.get(self._param_name, range(len(spectra)))
        assert len(spectra) == len(params)
        self.spectra = spectra
        self.params = params

    @classmethod
    def from_filenames(cls, fnames, zip_fname=None, **kwargs):
        spectra = [Spectrum.from_filename(fname, zip_fname=zip_fname) for fname in fnames]
        return cls(spectra=spectra, **kwargs)

    @classmethod
    def from_krx(cls, fname, **kwargs):
        spectra = Spectrum.from_krx(fname, page=None)
        s = spectra[0]
        if 'MapCoordinate' in s._metadata:
            assert len(spectra) == s['MapNoXSteps']
            kwargs.setdefault(cls._param_name,
                              np.linspace(s['MapStartX'], s['MapEndX'], s['MapNoXSteps']))
        return cls(spectra=spectra, **kwargs)

    def __getattr__(self, attr):
        if attr == self._param_name:
            return self.params
        raise AttributeError

    @property
    def data(self):
        return np.stack([s.data for s in self.spectra])

    def __array__(self):
        return self.data

    def _ipython_display_(self):
        from .widgets import isowidget

        display(isowidget(self))

    def plot(self, *args, **kwargs):
        pass

    def generate_fermimap(self, fl, width, dither_repair=False):
        fmap = []
        if isinstance(fl, int):
            fl = [fl] * len(self.spectra)
        for s, fl in zip(self.spectra, fl):
            fmap.append(s.data[-width + fl:fl + width].mean(axis=0))

        fmap = np.array(fmap)
        if dither_repair:
            invalid_area = fmap < 0.1*np.median(fmap)
            invalid_area[:, np.average(invalid_area, axis=0) > 0.9] = True
            fmap_ma = np.ma.masked_where(invalid_area, fmap)
            lens_profile = fmap_ma.sum(axis=0)
            fmap = fmap * gaussian_filter(lens_profile, 40)/lens_profile
        return fmap

class AngleMap(SpectrumMap):
    _param_name = 'angles'
    _param_label = 'Angle / deg'

    def plot(self, ax, lens_angle_c=1., other_angle_c=1., **kwargs):
        try:
            fmap = kwargs.pop('fmap')
        except KeyError:
            fmap = self.generate_fermimap(
                kwargs.pop('fl'), kwargs.pop('width', 10), kwargs.pop('dither_repair', False))

        # (-0.5, numcols-0.5, numrows-0.5, -0.5)
        s = self.spectra[0]
        da = (self.angles[-1] - self.angles[0])/len(self.angles)
        dl = (s.lens_scale[-1] - s.lens_scale[0])/len(s.lens_scale)
        extent = [lens_angle_c * (s.lens_scale[0] - dl/2),
                  lens_angle_c * (s.lens_scale[-1] + dl/2),
                  other_angle_c * (self.angles[0] - da/2),
                  other_angle_c * (self.angles[-1] + da/2)]
        kwargs.setdefault('extent', extent)
        kwargs.setdefault('cmap', 'inferno')
        ax.set_xlabel('Lens angle / deg')
        ax.set_ylabel('Deflection angle / deg')
        return ax.imshow(fmap, origin='lower', **kwargs)

    def plot_k(self, ax, fl, width=10, lens_angle_c=1., other_angle_c=1., new_origin=None, **kwargs):
        X = self.spectra[len(self.spectra) // 2].lens_scale * lens_angle_c
        Y = self.angles * other_angle_c
        X, Y = np.meshgrid(X, Y)

        e_fl = self.spectra[len(self.spectra) // 2].energy_scale[fl]
        X2 = np.sqrt(e_fl - 4) * 0.512 * np.sin(np.radians(X))  # + np.cos(np.radians(30))*Y*5.068*10**-4
        Y2 = np.sqrt(e_fl - 4) * 0.512 * np.sin(np.radians(Y))  # + np.cos(np.radians(30))*Y*5.068*10**-4

        try:
            fmap = kwargs.pop('fmap')
        except KeyError:
            fmap = self.generate_fermimap(
                fl, width, kwargs.pop('dither_repair', False))

        if new_origin:
            X2 = X2 - new_origin[0]
            Y2 = Y2 - new_origin[1]

        kwargs.setdefault('cmap', 'gist_yarg')
        kwargs.setdefault('shading', 'gouraud')
        kwargs.setdefault('rasterized', True)
        ax.pcolormesh(X2, Y2, fmap, **kwargs)
        ax.set_xlabel(r'$k_\parallel^\mathrm{Lens}$ / $1/\mathrm{\AA}$')
        ax.set_ylabel(r'$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$')

class DeflectionMap(AngleMap):
    _param_name = 'angles'
    _param_label = 'Deflection angle / deg'

class EnergyMap(SpectrumMap):
    _param_name = 'energies'
    _param_label = 'Photon Energy / deg'

    @property
    def fls(self):
        return [fl_guess(np.arange(len(s.energy_scale)), s.edc) for s in self.spectra]

    def fls_fit(self, fls=None, order=2):
        fls = fls or self.fls
        p = np.poly1d(np.polyfit(self.energies, fls, order))
        return np.around(p(self.energies)).astype(int)

    @classmethod
    def get_coord_transformer(cls, V_0=0, photon_angle=30, WF=4., BE=0., ):
        """Returns a function that transforms coordinate pairs (phi, hv) to (kx, kz)
        V_0 (eV): inner potential relative to E_vac
            V_0 (rel. to E_F) = V_0 (rel. to E_vac) - WF
        photon_angle (deg): angle away from grazing incidence (assuming normal emission), 
            i.e. (90 - photon_angle) == angle from normal incidence 
        WF (eV): sample work function - usually unknown!
        BE (eV): binding energy, >0

        If you assume that WF_sample == WF_detector, then hv-WF-BE is essentially the kinetic energy you see on the detector.

        # 0.5124 == ...
        # 5.067*10**-4 == ...
        """
        def transform(phi, hv):
            # see e.g. dx.doi.org/10.1107/S1600577513019085
            # todo ky != 0, unify all k-space transformers
            kz = np.sqrt((hv-WF-BE)*np.cos(np.deg2rad(phi))**2+V_0)*0.5124 + np.sin(np.deg2rad(photon_angle))*hv*5.067*10**-4
            kx = np.sqrt(hv-WF-BE)*0.5124*np.sin(np.deg2rad(phi)) #+ np.cos(np.deg2rad(30))*Y*5.068*10**-4
            return kx, kz
        return transform

    def plot_k(self, ax, fmap=None, lens_angle_c=1., angle_zero=0, tf_kwargs={}, **kwargs):
        phi = lens_angle_c * (self.spectra[0].lens_scale - angle_zero)
        hv = self.energies
        iso_cut = fmap

        phi, hv = np.meshgrid(phi, hv)

        tf = self.get_coord_transformer(**tf_kwargs)
        kx, kz = tf(phi, hv)
        kwargs.setdefault('cmap', 'inferno')
        kwargs.setdefault('shading', 'gouraud')
        kwargs.setdefault('rasterized', True)
        pc = ax.pcolormesh(kx, kz, iso_cut, **kwargs)
        ax.set_ylabel(r'$k_\perp$ / $\mathrm{\AA}^{-1}$')
        ax.set_xlabel(r'$k_\parallel$ / $\mathrm{\AA}^{-1}$')
        ax.set_aspect('equal')
        return pc