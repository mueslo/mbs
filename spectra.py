import operator
from functools import reduce
from os.path import splitext

from .load import parse_data, parse_info

import numpy as np

class Spectrum(object):
    def __init__(self, data, metadata):
        self._data = data
        self._metadata = metadata
        self._path = None

    @classmethod
    def from_filename(cls, fname, zip_fname=None):
        spec = cls(*parse_data(fname, zip_fname=zip_fname))
        spec._path = fname, zip_fname
        return spec

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

    def __getitem__(self, item):
        return self._metadata[item]

    @property
    def lens_extent(self):
        return self['XScaleMin'], self['XScaleMax']

    @property
    def lens_scale(self):
        return np.linspace(self['XScaleMin'], self['XScaleMax'], self['NoS'])

    @property
    def energy_extent(self):
        return self['Start K.E.'], self['End K.E.']

    @property
    def energy_scale(self):
        return np.linspace(self["Start K.E."],
                           self["End K.E."] - self['Step Size'],
                           len(self._data))
        #return np.arange(self['Start K.E.'], self['End K.E.'], self['Step Size'])

    @property
    def name(self):
        return self['Gen. Name']

    def __add__(self, other):
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
            return Exception('Operation not supported {}+{}'.format(type(self), type(other)))

        return SpectrumSum(self._data + other._data, m + om)

    def x_to_idx(self, x):
        pass

    def y_to_idx(self, y):
        pass

    def plot(self, ax, angle_correction=1., **kwargs):
        extent = self.lens_extent + self.energy_extent
        kwargs.setdefault('cmap', 'gist_yarg')
        kwargs.setdefault('vmin', np.percentile(self._data, 5))
        kwargs.setdefault('vmax', np.percentile(self._data, 99.5))
        kwargs.setdefault('aspect', (extent[1] - extent[0]) / (extent[3] - extent[2]))
        im = ax.imshow(self._data, origin='lower', extent=extent, **kwargs)
        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.set_xlabel(self['XScaleName'])
        return im

    def plot_edc(self, ax, e_f=None, **kwargs):
        show_counts = kwargs.pop('show_counts', False)
        if e_f is not None:
            x_scale = e_f - self.energy_scale
            xlabel = r'$E_\mathrm{bind}$ / eV'
            ax.invert_xaxis()
        else:
            x_scale = self.energy_scale
            xlabel = r'$E_\mathrm{kin}$ / eV'
        lines = ax.plot(x_scale, np.sum(self._data, axis=1), **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Intensity')
        if not show_counts:
            ax.set_yticks([], [])
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
        kwargs.setdefault('vmin', np.percentile(self._data, 5))
        kwargs.setdefault('vmax', np.percentile(self._data, 99.5))

        if k_origin:
            X2 = X2 - k_origin

        ax.set_ylabel(r'$E_\mathrm{kin}$ / eV')
        ax.set_xlabel(r'$k_\parallel$ / $1/\AA$')

        if Ef:
            Y2 = Y2 - Ef
            ax.set_ylabel(r'$E-E_\mathrm{F}$ / eV')

        im = ax.pcolormesh(X2, Y2, self._data,
                           shading='gouraud', **kwargs)
        return im

    def get_focus(self):
        assert not self['Lens Mode'].startswith('L4Ang')
        focus = np.sum(self._data, axis=0)
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

    def __getitem__(self, item):
        vals = [m[item] for m in self._metadata]
        if not vals or vals.count(vals[0]) == len(vals):
            return vals[0]
        else:
            return vals

    @property
    def name(self):
        return 'Sum of ' + ', '.join(self['Gen. Name'])


class Fermimap(object):
    def __init__(self, spectra, angles):
        self.spectra = spectra
        self.angles = angles

    @classmethod
    def from_filenames(cls, fnames, angles, zip_fname=None):
        return cls(
            spectra=[Spectrum(*parse_data(fname, zip_fname=zip_fname)) for fname in fnames],
            angles=angles)

    def generate_fermimap(self, fl, width):
        fmap = []
        for s in self.spectra:
            fmap.append(np.average(s._data[-width + fl:fl + width], axis=0))
        return np.array(fmap)

    def plot(self, ax, fl, width=10, lens_angle_c=1., other_angle_c=1., **kwargs):
        fmap = self.generate_fermimap(fl, width)
        # (-0.5, numcols-0.5, numrows-0.5, -0.5)
        extent = [lens_angle_c * self.spectra[0]['XScaleMin'],
                  lens_angle_c * self.spectra[0]['XScaleMax'],
                  other_angle_c * self.angles[0],
                  other_angle_c * self.angles[-1]]
        kwargs.setdefault('cmap', 'inferno')
        ax.imshow(fmap, origin='lower',
                  extent=extent, **kwargs)
        ax.set_xlabel('Lens angle / deg')
        ax.set_ylabel('Deflection angle / deg')

    def plot_k(self, ax, fl, width=10, lens_angle_c=1., other_angle_c=1., new_origin=None, **kwargs):
        X = self.spectra[len(self.spectra) // 2].lens_scale * lens_angle_c
        Y = self.angles * other_angle_c
        X, Y = np.meshgrid(X, Y)

        e_fl = self.spectra[len(self.spectra) // 2].energy_scale[fl]
        X2 = np.sqrt(e_fl - 4) * 0.512 * np.sin(np.radians(X))  # + np.cos(np.radians(30))*Y*5.068*10**-4
        Y2 = np.sqrt(e_fl - 4) * 0.512 * np.sin(np.radians(Y))  # + np.cos(np.radians(30))*Y*5.068*10**-4
        fmap = self.generate_fermimap(fl, width)

        if new_origin:
            X2 = X2 - new_origin[0]
            Y2 = Y2 - new_origin[1]

        kwargs.setdefault('cmap', 'gist_yarg')
        kwargs.setdefault('shading', 'gouraud')
        ax.pcolormesh(X2, Y2, fmap, **kwargs)
        ax.set_xlabel(r'$k_\parallel^\mathrm{Lens}$ / $1/\mathrm{\AA}$')
        ax.set_ylabel(r'$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$')
