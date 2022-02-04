import operator
from functools import reduce
from os.path import splitext
from collections import namedtuple
from enum import Enum
from copy import copy as shallow_copy

from .io import parse_data, parse_lines, parse_info, frame_unit
from .krx import KRXFile
from .utils import fl_guess

import numpy as np
from scipy.ndimage import gaussian_filter


scale = namedtuple('Scale', ['min', 'max', 'step', 'name'])
AcqMode = Enum('AcquisitionMode', 'Fixed Swept Dither')

class Spectrum(object):
    def __init__(self, data, metadata):
        self._data = data
        self._view = []  # iterative slices of original array
        self._metadata = metadata
        self._path = None

        if self.acq_mode == AcqMode.Dither:
            self._view.append(
                (slice(0, -self['DithSteps']),) + (slice(None),) * (data.ndim - 1))

    @classmethod
    def from_filename(cls, fname, zip_fname=None):
        spec = cls(*parse_data(fname, zip_fname=zip_fname))
        spec._path = fname, zip_fname
        return spec

    @classmethod
    def from_krx(cls, fname, page=0):
        kf = KRXFile(fname)
        metadata = parse_lines(
            kf.page_metadata(page).splitlines(),
            metadata_only=True)
        spec = cls(data=kf.page(page), metadata=metadata)
        spec._path = fname, None
        return spec

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

        fname, zip_fname = self._path
        info_path = splitext(fname)[0] + '.info'
        try:
            return parse_info(info_path, zip_fname)
        except IOError:
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

    def _translate_slice(self, slicetuple):
        slice_energy, slice_lens = slicetuple
        # todo: slice(None)
        return (slice(*list(map(self.e_to_i, [slice_energy.start, slice_energy.stop]))),
                slice(*list(map(self.l_to_i, [slice_lens.start, slice_lens.stop]))))

    def get_metadata(self, key):
        return self._metadata[key]

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
        if isinstance(other, int) and other == 0:
            return self

        # assert angle/energy extent is the same
        assert (self.lens_scale == other.lens_scale).all()
        assert (self.energy_scale == other.energy_scale).all()
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
        return np.sum(self.data, axis=1)

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
        if norm == 'max':
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

    def get_metadata(self, item):
        vals = [m[item] for m in self._metadata]
        if not vals or vals.count(vals[0]) == len(vals):
            return vals[0]
        else:
            return vals

    @property
    def name(self):
        return 'Sum of ' + ', '.join(self['Gen. Name'])


class SpectrumMap(object):  # 1D parameter space for now
    _param_name = 'params'
    def __init__(self, spectra, **kwargs):
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
        kf = KRXFile(fname)
        spectra = [Spectrum.from_krx(fname, i) for i in range(kf.num_pages)]
        s = spectra[0]
        if 'MapCoordinate' in s._metadata:
            assert kf.num_pages == s['MapNoXSteps']
            kwargs.setdefault(cls._param_name,
                              np.linspace(s['MapStartX'], s['MapEndX'], s['MapNoXSteps']))
        return cls(spectra=spectra, **kwargs)

    def __getattr__(self, attr):
        if attr == self._param_name:
            return self.params
        raise AttributeError

    @property
    def array(self):
        return np.stack([s.data for s in self.spectra])

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
        def transform(phi, hv):
            # dx.doi.org/10.1107/S1600577513019085
            kz = np.sqrt((hv-WF-BE)*np.cos(np.radians(phi))**2+V_0)*0.5124 + np.sin(np.radians(photon_angle))*hv*5.067*10**-4
            kx = np.sqrt(hv-WF-BE)*0.5124*np.sin(np.radians(phi)) #+ np.cos(np.radians(30))*Y*5.068*10**-4
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