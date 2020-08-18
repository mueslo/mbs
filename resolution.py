import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from .load import parse_data


k_B = 8.617333262145 * 10 ** -5  # eV/K


def F(E, E_f, T):
    return 1/(1+np.exp((E-E_f)/(k_B*T)))


def fermiedge(E, a, b, C, E_f, T):
    return F(E, E_f, T)*(a*(E-E_f) + b) + C


def make_fe(T, step_size):
    def broadened_fermiedge(E, a, b, C, sigma, E_f):
        return gaussian_filter1d(fermiedge(E, a, b, C, E_f, T), sigma/step_size)
    return broadened_fermiedge


def cut(x, y, window):
    idx_start = (np.abs(x - window[0])).argmin()
    idx_end = (np.abs(x - window[1])).argmin()
    return x[idx_start:idx_end], y[idx_start:idx_end]


def fermi_fit(data, metadata, T, de=1., ax=None):
    edc = data.sum(axis=1)
    e_ax = np.linspace(metadata['Start K.E.'], metadata['End K.E.'], len(edc))
    fl = e_ax[np.max(np.argwhere(edc > np.percentile(edc, 50)))]
    window = (fl - de / 2, fl + de / 2)
    cut_e_ax, cut_edc = cut(e_ax, edc, window)

    func = make_fe(T, metadata['Step Size'])

    initial_guess = (0, max(cut_edc), min(edc), 0.05, fl)
    bounds = [(-np.inf, -np.inf, 0, 0, window[0]),
              (np.inf, np.inf, np.inf, np.inf, window[1])]
    fit_params, cov = curve_fit(func, cut_e_ax, cut_edc,
                                p0=initial_guess,
                                bounds=bounds)
    fit_params_d = dict(zip(["a", "b", "C", "sigma", "E_f"], fit_params))
    fit_params_d['factor'] = 1.
    fit_params_d['fwhm'] = 2.3548 * fit_params_d['sigma']

    if ax:
        ax.plot(e_ax, edc, linewidth=0.5)
        ax.plot(cut_e_ax, func(cut_e_ax, *fit_params))
        ax.set_xlim(cut_e_ax[0], cut_e_ax[-1])
        # ax.plot(cut_e_ax, func(cut_e_ax, *initial_guess), color='k')

    return fit_params_d


def opt_global(fits, T, *parsed_data, de=1., plot_fits=False):
    func = make_fe(T, parsed_data[0][1]['Step Size']) # todo implement separate global err function for each

    def global_err(p, *xys, verbose=False):
        assert len(xys) % 2 == 0 and len(xys) > 0
        assert len(p) == 3 + 3 * (len(xys)//2)

        errs = []
        for i in range(len(xys)//2):
            factor = p[3+3*i] # factor
            p_i = p[0], p[1], p[2], p[3+3*i+1], p[3+3*i+2] # a, b, C, sigma, E_f
            x, y = xys[2*i:2*i+2]
            errs.extend(factor*func(x, *p_i) - y)
            errs.append(1000*(factor-1))
            if verbose:
                print("subfit", i)
                print(factor)
                print(p_i)

        return np.array(errs)

    a_global = np.mean([fit["a"] for fit in fits])
    b_global = np.mean([fit["b"] for fit in fits])
    C_global = np.mean([fit["C"] for fit in fits])
    p_global = [a_global, b_global, C_global]
    bounds_g = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]

    xys = []
    for i, fit in enumerate(fits):
        p_global.append(1) # factor
        bounds_g.append((0, np.inf))

        p_global.append(fit["sigma"]*10)
        bounds_g.append((0, np.inf))

        p_global.append(fit["E_f"])
        edc = parsed_data[i][0].sum(axis=1)
        e_ax = np.linspace(parsed_data[i][1]['Start K.E.'], parsed_data[i][1]['End K.E.'], len(edc))
        fl = fit["E_f"]
        window = (fl - de / 2, fl + de / 2)
        bounds_g.append(window)
        xys.extend(cut(e_ax, edc, window))

    result = least_squares(global_err, p_global, args=xys, bounds=np.array(bounds_g).T)
    if not result.success:
        print("global fit result", result)
        raise Exception('global fit failed')
    else:
        p_best = result.x

    fits_global = []
    for i, p in enumerate(fits):
        factor = p_best[3+3*i] # factor
        p_i = p_best[0], p_best[1], p_best[2], p_best[3+3*i+1], p_best[3+3*i+2] # a, b, C, sigma, E_f
        x, y = xys[2*i:2*i+2]

        #print('factor', repr(factor))
        fit = dict(zip(["factor", "a", "b", "C", "sigma", "E_f"], (factor,)+p_i))
        fit['fwhm'] = 2.3548 * fit['sigma']
        fits_global.append(fit)
        #print(fit)
        if plot_fits:
            plt.figure(figsize=(3, 1))
            plt.plot(np.linspace(parsed_data[i][1]['Start K.E.'],
                                 parsed_data[i][1]['End K.E.'], len(edc)), parsed_data[i][0].sum(axis=1))
            plt.plot(x, factor*func(x, *p_i))
            plt.xlim(x[0], x[-1])
            plt.show()

    return fits_global


def plot_opt(paths, params, T, de=1., param_name='measurement param', plot_param="sigma", fit_global=False, plot_fits=False,
             zip_fname=None, label=None):
    dmd = []
    fits = []
    for p in paths:
        data, metadata = parse_data(p, zip_fname=zip_fname)
        data = data[:, 1:]
        dmd.append((data, metadata))
        if plot_fits:
            plt.figure(figsize=(3, 1))
            fit = fermi_fit(data, metadata, T=T, de=de, ax=plt.gca())
            plt.show()
        else:
            fit = fermi_fit(data, metadata, T=T, de=de)
        fits.append(fit)

    plt.plot(params, [fit[plot_param] for fit in fits], 'x', label=label or 'individual fit')
    plt.xlabel(param_name)
    plt.ylabel(plot_param)

    if fit_global:
        fits_global = opt_global(fits, T, *dmd, de=de, plot_fits=plot_fits)
        plt.plot(params, [fit[plot_param] for fit in fits_global], 'x', label=label or 'global fit')
        plt.legend()
        return fits, fits_global

    return fits