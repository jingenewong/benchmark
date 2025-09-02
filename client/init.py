import random, numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import tiktoken
import re
import string

import matplotlib
matplotlib.use('Agg')

kB = 8.6173E-5     # Boltzmann constant (eV/K)
pi = np.pi





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Create blank spectra -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def create_spectrum(k_min, k_max, k_pnts, E_min, E_max, E_pnts, background):
    k_array = np.linspace(k_min, k_max, k_pnts, endpoint = True)
    E_array = np.linspace(E_min, E_max, E_pnts, endpoint = True)
    spectrum = np.ones((E_pnts, k_pnts))*background
    return k_array, E_array, spectrum



def create_map(kx_min, kx_max, kx_pnts, ky_min, ky_max, ky_pnts, background):
    kx_array = np.linspace(kx_min, kx_max, kx_pnts, endpoint = True)
    ky_array = np.linspace(ky_min, ky_max, ky_pnts, endpoint = True)
    map = np.ones((ky_pnts, kx_pnts))*background
    return kx_array, ky_array, map





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Plot spectra -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_spectrum(k_array, E_array, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_map(kx_array, ky_array, map, name):

    fig = plt.figure('Parallel', figsize = (8, 8), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(map, extent = (kx_array.min(), kx_array.max(), ky_array.min(), ky_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k_x$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Momentum $k_y$ ($\AA^{-1}$)')
    ax1.axis('equal')

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_map_trace(kx_array, ky_array, kx_trace, ky_trace, map, name):

    fig = plt.figure('Parallel', figsize = (8, 8), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(map, extent = (kx_array.min(), kx_array.max(), ky_array.min(), ky_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k_x$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Momentum $k_y$ ($\AA^{-1}$)')
    ax1.axis('equal')
    plt.scatter(kx_trace, ky_trace, s = 0.1, c = 'white')

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_fwhm(k_array, E_array, disp_array, fwhm_array, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(disp_array, E_array, 'w', linewidth = 0.5)
    plt.plot(disp_array - fwhm_array/2, E_array, 'w--', linewidth = 0.3)
    plt.plot(disp_array + fwhm_array/2, E_array, 'w--', linewidth = 0.3)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_trace_k(k_array, E_array, Ek_array, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_array, Ek_array, 'w', linewidth = 0.5)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_trace_E(k_array, E_array, disp_array, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(disp_array, E_array, 'w', linewidth = 0.5)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_trace_E_2(k_array, E_array, disp_array_1, disp_array_2, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(disp_array_1, E_array, 'w', linewidth = 0.5)
    plt.plot(disp_array_2, E_array, 'w', linewidth = 0.5)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_trace_gap(k_array, E_array, Delta, spectrum, name):

    Delta_array = np.ones(len(k_array))
    Delta_array_a = Delta*Delta_array
    Delta_array_b = -Delta*Delta_array

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_array, Delta_array_a, 'w--', linewidth = 0.5)
    plt.plot(k_array, Delta_array_b, 'w--', linewidth = 0.5)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_phonon(k_array, E_array, k_phonon, k_band, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_phonon, E_array, 'r', linewidth = 0.5)
    plt.plot(k_band, E_array, 'w--', linewidth = 0.3)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_spectrum_fwhm_numeric(k_array, E_array, k_band, k_hwhm_left, k_hwhm_right, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_band, E_array, 'r', linewidth = 0.5)
    plt.plot(k_hwhm_left, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_hwhm_right, E_array, 'r:', linewidth = 0.5)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_spectrum_fwhm_both(k_array, E_array, k_band, k_hwhm_left, k_hwhm_right, disp_array, fwhm_array, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_band, E_array, 'r', linewidth = 0.5)
    plt.plot(k_hwhm_left, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_hwhm_right, E_array, 'r:', linewidth = 0.5)
    plt.plot(disp_array, E_array, 'w', linewidth = 0.5)
    plt.plot(disp_array - fwhm_array/2, E_array, 'w--', linewidth = 0.3)
    plt.plot(disp_array + fwhm_array/2, E_array, 'w--', linewidth = 0.3)
    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_spectrum_1_phonon(k_array, E_array, k_phonon, k_band, k_hwhm_left, k_hwhm_right, freq, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_phonon, E_array, 'r', linewidth = 0.5)
    plt.plot(k_hwhm_left, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_hwhm_right, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_band, E_array, 'w--', linewidth = 0.3)

    phonon_array = np.ones(len(k_array))*freq
    plt.plot(k_array, phonon_array, 'w:', linewidth = 0.3)

    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_spectrum_2_phonons(k_array, E_array, k_phonon, k_band, k_hwhm_left, k_hwhm_right, freq1, freq2, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_phonon, E_array, 'r', linewidth = 0.5)
    plt.plot(k_hwhm_left, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_hwhm_right, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_band, E_array, 'w--', linewidth = 0.3)

    phonon_array_1 = np.ones(len(k_array))*freq1
    plt.plot(k_array, phonon_array_1, 'w:', linewidth = 0.3)
    phonon_array_2 = np.ones(len(k_array))*freq2
    plt.plot(k_array, phonon_array_2, 'w:', linewidth = 0.3)

    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_spectrum_3_phonons(k_array, E_array, k_phonon, k_band, k_hwhm_left, k_hwhm_right, freq1, freq2, freq3, spectrum, name):

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    ax1 = fig.add_subplot(111)
    ax1.imshow(spectrum, extent = (k_array.min(), k_array.max(), E_array.min(), E_array.max()), aspect = 'auto', cmap = 'viridis', interpolation = 'none', origin = 'lower')
    ax1.set_xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    ax1.set_ylabel(r'Energy $\omega$ (eV)')
    plt.plot(k_phonon, E_array, 'r', linewidth = 0.5)
    plt.plot(k_hwhm_left, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_hwhm_right, E_array, 'r:', linewidth = 0.5)
    plt.plot(k_band, E_array, 'w--', linewidth = 0.3)

    phonon_array_1 = np.ones(len(k_array))*freq1
    plt.plot(k_array, phonon_array_1, 'w:', linewidth = 0.3)
    phonon_array_2 = np.ones(len(k_array))*freq2
    plt.plot(k_array, phonon_array_2, 'w:', linewidth = 0.3)
    phonon_array_3 = np.ones(len(k_array))*freq3
    plt.plot(k_array, phonon_array_3, 'w:', linewidth = 0.3)

    plt.xlim(k_array.min(), k_array.max())
    plt.ylim(E_array.min(), E_array.max())

    full_path = f'/workspaces/physics-benchmark/client/{name}.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_fwhm(E_array, fwhm_array, k_hwhm_left, k_hwhm_right, mu, name):

    max_fwhm = max(np.max(np.nan_to_num(np.absolute(k_hwhm_right - k_hwhm_left))), np.max(fwhm_array))

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(E_array, np.absolute(k_hwhm_right - k_hwhm_left), 'r', linewidth = 1)
    plt.plot(E_array, np.absolute(fwhm_array), 'k--', linewidth = 1)
    plt.xlim(np.min(E_array), mu)
    plt.ylim(0, max_fwhm + 0.1)
    plt.xlabel(r'Energy $\omega$ (eV)')
    plt.ylabel(r'FWHM ($\AA^{-1}$)')

    full_path = f'/workspaces/physics-benchmark/client/{name}_FWHM.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_sigma_1_phonon(E_array, k_band, k_phonon, k_hwhm_left, k_hwhm_right, ReS_array, ImS_array, mu, conv, freq1, name):

    ReS = np.absolute(k_phonon - k_band)
    ImS = np.absolute(k_hwhm_right - k_hwhm_left)/2
    max_value = max(np.max(np.nan_to_num(ImS)), np.max(np.nan_to_num(ReS)))

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(E_array, ImS, 'r', linewidth = 1)
    plt.plot(E_array, ReS, 'b', linewidth = 1)
    plt.plot(E_array, ImS_array, 'r--', linewidth = 0.5)
    plt.plot(E_array, ReS_array, 'b--', linewidth = 0.5)

    plt.xlim(np.min(E_array), mu)
    plt.ylim(0, max_value + 0.1)
    plt.axvline(x = mu - freq1, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq1 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq1 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.xlabel(r'Energy $\omega$ (eV)')
    plt.ylabel(r'Re/Im $\Sigma(\omega)$')

    full_path = f'/workspaces/physics-benchmark/client/{name}_Sigma.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_sigma_2_phonons(E_array, k_band, k_phonon, k_hwhm_left, k_hwhm_right, ReS_array, ImS_array, mu, conv, freq1, freq2, name):

    ReS = np.absolute(k_phonon - k_band)
    ImS = np.absolute(k_hwhm_right - k_hwhm_left)/2
    max_value = max(np.max(np.nan_to_num(ImS)), np.max(np.nan_to_num(ReS)))

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(E_array, ImS, 'r', linewidth = 1)
    plt.plot(E_array, ReS, 'b', linewidth = 1)
    plt.plot(E_array, ImS_array, 'r--', linewidth = 0.5)
    plt.plot(E_array, ReS_array, 'b--', linewidth = 0.5)
    
    plt.xlim(np.min(E_array), mu)
    plt.ylim(0, max_value + 0.1)
    plt.axvline(x = mu - freq1, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq1 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq1 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq2, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq2 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq2 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.xlabel(r'Energy $\omega$ (eV)')
    plt.ylabel(r'Re/Im $\Sigma(\omega)$')

    full_path = f'/workspaces/physics-benchmark/client/{name}_Sigma.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def plot_sigma_3_phonons(E_array, k_band, k_phonon, k_hwhm_left, k_hwhm_right, ReS_array, ImS_array, mu, conv, freq1, freq2, freq3, name):

    ReS = np.absolute(k_phonon - k_band)
    ImS = np.absolute(k_hwhm_right - k_hwhm_left)/2
    max_value = max(np.max(np.nan_to_num(ImS)), np.max(np.nan_to_num(ReS)))

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(E_array, ImS, 'r', linewidth = 1)
    plt.plot(E_array, ReS, 'b', linewidth = 1)
    plt.plot(E_array, ImS_array, 'r--', linewidth = 0.5)
    plt.plot(E_array, ReS_array, 'b--', linewidth = 0.5)
    
    plt.xlim(np.min(E_array), mu)
    plt.ylim(0, max_value + 0.1)
    plt.axvline(x = mu - freq1, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq1 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq1 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq2, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq2 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq2 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq3, color = 'k', ls = '--', lw = 0.5)
    plt.axvline(x = mu - freq3 + conv, color = 'k', ls = ':', lw = 0.5)
    plt.axvline(x = mu - freq3 - conv, color = 'k', ls = ':', lw = 0.5)
    plt.xlabel(r'Energy $\omega$ (eV)')
    plt.ylabel(r'Re/Im $\Sigma(\omega)$')

    full_path = f'/workspaces/physics-benchmark/client/{name}_Sigma.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Add noise -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def add_noise_spectrum(k_array, E_array, spectrum, amplitude):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_noise = np.zeros((E_pnts, k_pnts))

    for E_num in range(E_pnts):
        for k_num in range(k_pnts):
            spectrum_noise[E_num, k_num] = random.random()*amplitude
    
    spectrum_new += spectrum_noise
    return spectrum_new



def add_noise_map(kx_array, ky_array, map, amplitude):

    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    map_new = map
    map_noise = np.zeros((ky_pnts, kx_pnts))
    
    for ky_num in range(ky_pnts):
        for kx_num in range(kx_pnts):
            map_noise[ky_num, kx_num] = random.random()*amplitude
    
    map_new += map_noise
    return map_new



def add_noise_scaled_spectrum(k_array, E_array, spectrum, noise_num, sigma_E, sigma_k, mu, T, E_conv, ratio):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_noise = np.zeros((E_pnts, k_pnts))

    for num in range(noise_num):
        mu_k = k_array[random.randrange(k_pnts)]
        mu_E = E_array[random.randrange(E_pnts)]
        exp_k = np.exp((-0.5)*((k_array - mu_k)/sigma_k)**2)
        exp_E = np.exp((-0.5)*((E_array - mu_E)/sigma_E)**2)
        spectrum_noise += (1/noise_num)*np.outer(exp_E, exp_k)/(2*pi*sigma_E*sigma_k)
    
    spectrum_noise_FD = multiply_FD(E_array, k_array, spectrum_noise, mu, T)
    spectrum_noise_convolve = convolve_E(k_array, E_array, spectrum_noise_FD, E_conv)
    spectrum_max = np.max(spectrum)
    spectrum_noise_integrated = np.mean(spectrum_noise_convolve)
    spectrum_new = spectrum + spectrum_noise_convolve*ratio*(spectrum_max/spectrum_noise_integrated)

    return spectrum_new



def add_noise_swept_spectrum(k_array, E_array, spectrum, noise_num, sigma_k, mu, T, E_sigma, ratio):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_noise = np.zeros((E_pnts, k_pnts))

    for num in range(noise_num):
        mu_k = k_array[random.randrange(k_pnts)]
        exp_k = np.exp((-0.5)*((k_array - mu_k)/sigma_k)**2)
        E_fill = np.ones(E_pnts)
        spectrum_noise += (1/noise_num)*np.outer(E_fill, exp_k)/(np.sqrt(2*pi)*sigma_k)
    
    spectrum_noise_FD = multiply_FD(E_array, k_array, spectrum_noise, mu, T)
    spectrum_noise_convolve = convolve_E(k_array, E_array, spectrum_noise_FD, E_sigma)
    spectrum_max = np.max(spectrum)
    spectrum_noise_integrated = np.mean(spectrum_noise_convolve)
    spectrum_new = spectrum + spectrum_noise_convolve*ratio*(spectrum_max/spectrum_noise_integrated)

    return spectrum_new



def add_noise_scaled_map(kx_array, ky_array, map, noise_num, sigma_k, ratio):

    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    map_noise = np.zeros((ky_pnts, kx_pnts))

    for num in range(noise_num):
        mu_kx = kx_array[random.randrange(kx_pnts)]
        mu_ky = ky_array[random.randrange(ky_pnts)]
        exp_kx = np.exp((-0.5)*((kx_array - mu_kx)/sigma_k)**2)
        exp_ky = np.exp((-0.5)*((ky_array - mu_ky)/sigma_k)**2)
        map_noise += (1/noise_num)*np.outer(exp_ky, exp_kx)/(2*pi*(sigma_k**2))
    
    map_max = np.max(map)
    map_noise_integrated = np.mean(map_noise)
    map_new = map + map_noise*ratio*(map_max/map_noise_integrated)

    return map_new





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------- Fermi–Dirac and processes -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def multiply_FD(E_array, k_array, spectrum, mu, T):

    E_pnts = len(E_array); k_pnts = len(k_array)
    spectrum_new = np.zeros((E_pnts, k_pnts))

    for E_num in range(E_pnts):
        E = E_array[E_num]; kBT = kB*T
        FD = 1/(1 + np.exp((E - mu)/kBT))
        MDC = spectrum[E_num, :]*FD
        spectrum_new[E_num, :] = MDC
    
    return spectrum_new



def convolve_k(k_array, E_array, spectrum, k_sigma):
    k_pnts = len(k_array); E_pnts = len(E_array)
    dk = (k_array[-1] - k_array[0])/(k_pnts - 1)
    sigma_val = k_sigma/dk
    spectrum_convolve = scipy.ndimage.gaussian_filter(spectrum, sigma = sigma_val, axes = 1, mode = 'nearest')
    return spectrum_convolve



def convolve_E(k_array, E_array, spectrum, E_sigma):
    k_pnts = len(k_array); E_pnts = len(E_array)
    dE = (E_array[-1] - E_array[0])/(E_pnts - 1)
    sigma_val = E_sigma/dE
    spectrum_convolve = scipy.ndimage.gaussian_filter(spectrum, sigma = sigma_val, axes = 0, mode = 'nearest')
    return spectrum_convolve



def convolve_kxky(kx_array, ky_array, map, k_sigma):
    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    dkx = (kx_array[-1] - kx_array[0])/(kx_pnts - 1)
    dky = (ky_array[-1] - ky_array[0])/(ky_pnts - 1)
    sigma_x_val = k_sigma/dkx
    sigma_y_val = k_sigma/dky
    map_convolve = scipy.ndimage.gaussian_filter(map, sigma = (sigma_y_val, sigma_x_val), axes = (0, 1), mode = 'nearest')
    return map_convolve



def symmetrise(spectrum):
    spectrum_new = spectrum
    spectrum_inverse = spectrum[::-1,:]
    return (spectrum_new + spectrum_inverse)/2





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------ Miscellanea -----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_hwhm(k_array, MDC_entry, peak):

    first_index = next(x for x, val in enumerate(MDC_entry)
        if val > peak/2)
    last_index = -1

    for i in range(len(MDC_entry)):
        if MDC_entry[i] > peak/2:
            last_index = i
    
    return k_array[first_index], k_array[last_index]



def nearest_k(array, value):
    return (np.abs(array - value)).argmin()



def array2string(array):
    return '\n'.join(['\t'.join(map(str, row)) for row in array])



def cut_spectrum(E_array, spectrum_raw, mu):
    index = next(x for x, val in enumerate(E_array)
        if val > mu)
    spectrum_cut = spectrum_raw[:index + 1,:]
    return spectrum_cut



def cut_array(E_array, array_raw, mu):
    index = next(x for x, val in enumerate(E_array)
        if val > mu)
    array_cut = array_raw[:index + 1]
    return array_cut



def cut_Ek_array(Ek_array, mu):
    new_array = Ek_array.astype(float)
    new_array[new_array > mu] = np.nan
    return new_array



def spectrum_dynamic_range(spectrum, dynamic_range):
    spectrum_max = np.max(spectrum)
    spectrum_rescaled = spectrum*dynamic_range/spectrum_max
    spectrum_integer = np.round(spectrum_rescaled).astype(int)
    spectrum_trunc = [[int(str(x).split('.')[0]) for x in row] for row in spectrum_integer]
    return spectrum_trunc



def convert_spectrum_to_txt(k_array, E_array, spectrum):
    temp = np.array(E_array)[:, np.newaxis]
    data = array2string(np.concatenate((np.round(temp, decimals = 3), np.round(spectrum, decimals = 3)), axis = 1))
    txt = "Energy (eV) / Momentum:\t" + '\t'.join(map(str, np.round(k_array, decimals = 4))) + "\n" + data
    txt_stripped = re.sub(r"\.0\b", "", txt)
    return txt_stripped



def convert_map_to_txt(kx_array, ky_array, map_data):
    temp = np.array(ky_array)[:, np.newaxis]
    data = array2string(np.concatenate((np.round(temp, decimals = 3), np.round(map_data, decimals = 3)), axis = 1))
    txt = "Momentum (y) / Momentum (x):\t" + '\t'.join(map(str, np.round(kx_array, decimals = 4))) + "\n" + data
    txt_stripped = re.sub(r"\.0\b", "", txt)
    return txt_stripped



def token_counter(txt):
    encoding = tiktoken.get_encoding("o200k_base")
    token_count = len(encoding.encode(txt))
    return token_count



def write_to_text(content, path, file_name):
  full_path = f'{path}{file_name}.txt'
  file = open(full_path, "w")
  file.write(content)
  file.close()
  f = open(full_path, "r")
  return



def score_array(ground_truth, response, sigma, factor):

    array_diff = ground_truth - response
    arr = np.absolute(array_diff)
    distance = sum(arr)/len(arr)
    counter = 0

    if distance < factor*sigma:
        counter = 1
    
    return counter





# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Bands (MDC) -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def add_band_MDC(k_array, E_array, vF, spectrum, disp_array, fwhm_array, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for E_num in range(E_pnts):
        fwhm = fwhm_array[E_num]
        peak = disp_array[E_num]
        sigma = fwhm/2

        for k_num in range(k_pnts):
            k = k_array[k_num]
            #spectrum_band[E_num, k_num] = intensity*sigma/(pi*(sigma**2 + (vF*(k - peak))**2))
            spectrum_band[E_num, k_num] = intensity*sigma/(sigma**2 + ((k - peak)**2))
    
    spectrum_new += spectrum_band
    return spectrum_new



def create_fwhm_constant(E_array, fwhm_value):
    E_pnts = len(E_array)
    fwhm_array = np.ones(E_pnts)*fwhm_value
    return fwhm_array



def create_fwhm_power_law(E_array, mu, a, n, delta):

    E_pnts = len(E_array)
    fwhm_array = np.zeros(E_pnts)

    for E_num in range(E_pnts):
        fwhm_array[E_num] = a*(np.absolute(E_array[E_num] - mu))**n + delta
    
    return fwhm_array



def create_fwhm_erf(E_array, mu, a, freq, scale, delta):

    E_pnts = len(E_array)
    fwhm_array = np.zeros(E_pnts)

    for E_num in range(E_pnts):
        fwhm_array[E_num] = a*(1 + scipy.special.erf((mu - E_array[E_num] - freq)/scale))/2 + delta
    
    return fwhm_array





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Bands (QP) -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def add_band_qp_linear(k_array, E_array, spectrum, vF, kF, mu, gamma_array, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for E_num in range(E_pnts):
        E = E_array[E_num]
        gamma = gamma_array[E_num]

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            spectrum_band[E_num, k_num] = 0.5*intensity*gamma/((E - Ek)**2 + (gamma/2)**2)
    
    spectrum_new += spectrum_band
    return spectrum_new



def add_band_qp_quadratic(k_array, E_array, spectrum, vF, kF, alpha, mu, gamma_array, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for E_num in range(E_pnts):
        E = E_array[E_num]
        gamma = gamma_array[E_num]

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_quadratic(k, vF, kF, alpha, mu)
            spectrum_band[E_num, k_num] = 0.5*intensity*gamma/((E - Ek)**2 + (gamma/2)**2)
    
    spectrum_new += spectrum_band
    return spectrum_new



def add_band_qp_band_bottom(k_array, E_array, spectrum, E0, k0, alpha, gamma_array_k, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for k_num in range(k_pnts):
        k = k_array[k_num]
        gamma = gamma_array_k[k_num]
        Ek = dispersion_band_bottom(k, E0, k0, alpha)

        for E_num in range(E_pnts):
            E = E_array[E_num]
            spectrum_band[E_num, k_num] = 0.5*intensity*gamma/((E - Ek)**2 + (gamma/2)**2)
    
    spectrum_new += spectrum_band
    return spectrum_new



def add_band_BCS_linear(k_array, E_array, spectrum, vF, kF, Gamma_0, Gamma_1, Delta, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for k_num in range(k_pnts):
        k = k_array[k_num]
        Ek = dispersion_linear(k, vF, kF, 0)

        for E_num in range(E_pnts):
            E = E_array[E_num]
            Sigma = -1j*Gamma_1 + (Delta**2)/(E + Ek + 1j*Gamma_0)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] += -(1/pi)*G.imag
    
    spectrum_new += spectrum_band
    return spectrum_new



def add_band_BCS_quadratic(k_array, E_array, spectrum, vF, kF, alpha, Gamma_0, Gamma_1, Delta, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for k_num in range(k_pnts):
        k = k_array[k_num]
        Ek = dispersion_quadratic(k, vF, kF, alpha, 0)

        for E_num in range(E_pnts):
            E = E_array[E_num]
            Sigma = -1j*Gamma_1 + (Delta**2)/(E + Ek + 1j*Gamma_0)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] += -(1/pi)*G.imag
    
    spectrum_new += spectrum_band
    return spectrum_new



def add_band_BCS_band_bottom(k_array, E_array, spectrum, E0, k0, alpha, Gamma_0, Gamma_1, Delta, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_new = spectrum
    spectrum_band = np.zeros((E_pnts, k_pnts))

    for k_num in range(k_pnts):
        k = k_array[k_num]
        Ek = dispersion_band_bottom(k, E0, k0, alpha)

        for E_num in range(E_pnts):
            E = E_array[E_num]
            Sigma = -1j*Gamma_1 + (Delta**2)/(E + Ek + 1j*Gamma_0)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] += -(1/pi)*G.imag
    
    spectrum_new += spectrum_band
    return spectrum_new



def create_dispersion_linear(E_array, vF, kF, mu):

    E_pnts = len(E_array)
    disp_array = np.zeros(E_pnts)

    for E_num in range(E_pnts):
        Ek = E_array[E_num]
        k = (Ek - mu)/vF + kF
        disp_array[E_num] = k
    
    return disp_array



def dispersion_linear(k, vF, kF, mu):
    Ek = mu + vF*(k - kF)
    return Ek



def create_dispersion_quadratic(E_array, vF, kF, alpha, mu):

    E_pnts = len(E_array)
    disp_array = np.zeros(E_pnts)
    a = kF - vF/(2*alpha)

    for E_num in range(E_pnts):
        Ek = E_array[E_num]
        b = np.sqrt(np.absolute((Ek - mu)/alpha + (vF**2)/(4*(alpha**2))))
        if vF/alpha > 0:
            k = a + b
        else:
            k = a - b
        disp_array[E_num] = k
    
    return disp_array



def dispersion_quadratic(k, vF, kF, alpha, mu):
    Ek = mu + vF*(k - kF) + alpha*((k - kF)**2)
    return Ek



def create_dispersion_band_bottom(k_array, E0, k0, alpha):

    k_pnts = len(k_array)
    E_array = np.zeros(k_pnts)

    for k_num in range(k_pnts):
        k = k_array[k_num]
        E = dispersion_band_bottom(k, E0, k0, alpha)
        E_array[k_num] = E
    
    return E_array



def dispersion_band_bottom(k, E0, k0, alpha):
    Ek = E0 + alpha*((k - k0)**2)
    return Ek



def get_Ek_band_bottom(k_array, band_bottom_params):

    E0 = band_bottom_params[0]; k0 = band_bottom_params[1]; alpha = band_bottom_params[2]
    k_pnts = len(k_array)
    Ek_band = np.zeros(k_pnts)

    for k_num in range(k_pnts):
        k = k_array[k_num]
        Ek_band[k_num] = dispersion_band_bottom(k, E0, k0, alpha)
    
    return Ek_band



def create_gamma_FL(E_array, mu, intensity, T):

    E_pnts = len(E_array)
    gamma_FL_array = np.zeros(E_pnts)

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Im_sigma = intensity*((E - mu)**2 + (pi*kB*T)**2)
        gamma_FL_array[E_num] = 2*Im_sigma
    
    return gamma_FL_array



def create_gamma_MFL(E_array, mu, intensity, T):

    E_pnts = len(E_array)
    gamma_FL_array = np.zeros(E_pnts)

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Im_sigma = intensity*(np.absolute(E - mu) + pi*kB*T)
        gamma_FL_array[E_num] = 2*Im_sigma
    
    return gamma_FL_array



def get_fwhm_array(E_array, k_array, mu, spectrum, k_int_range, gamma_array):

    E_pnts = len(E_array)
    k_band = np.zeros(E_pnts)
    k_hwhm_left = np.zeros(E_pnts)
    k_hwhm_right = np.zeros(E_pnts)
    gamma_array_trunc = gamma_array

    for E_num in range(E_pnts):
        E = E_array[E_num]

        if E < mu:
            peak_index = int(np.argmax(spectrum[E_num,:]))
            k_band[E_num] = k_array[peak_index]

            MDC = spectrum[E_num,:]
            k_min = int(nearest_k(k_array, k_int_range[0]))
            k_max = int(nearest_k(k_array, k_int_range[1]))
            sub_MDC = MDC[k_min:k_max]
            background = sum(sub_MDC)/len(sub_MDC)
            peak = MDC[peak_index] - background
            MDC_new = MDC - background*np.ones(len(k_array))

            hwhm_left, hwhm_right = get_hwhm(k_array, MDC_new, peak)
            k_hwhm_left[E_num] = hwhm_left
            k_hwhm_right[E_num] = hwhm_right
        else:
            k_band[E_num] = np.nan
            k_hwhm_left[E_num] = np.nan
            k_hwhm_right[E_num] = np.nan
            gamma_array_trunc[E_num] = np.nan
    
    return gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right



def F_function(x):
    F = -(1/3)*(x**3)*np.log(1 - x**(-2)) - (x/3) + (1/3)*np.log((1 - x)/(1 + x))
    return F



def add_band_phonon_linear(k_array, E_array, spectrum, vF, kF, mu, coupling_lambda, freq, delta, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma = coupling_lambda*freq*F_function((E - mu - delta*1j)/freq)

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array



def add_band_phonon_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda, freq, delta, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma = coupling_lambda*freq*F_function((E - mu - delta*1j)/freq) + 1j*gamma_array[E_num]/2

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array



def add_band_2_phonons_linear(k_array, E_array, spectrum, vF, kF, mu, coupling_lambda1, freq1, delta1, coupling_lambda2, freq2, delta2, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma_1 = coupling_lambda1*freq1*F_function((E - mu - delta1*1j)/freq1)
        Sigma_2 = coupling_lambda2*freq2*F_function((E - mu - delta2*1j)/freq2)
        Sigma = Sigma_1 + Sigma_2

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array



def add_band_2_phonons_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda1, freq1, delta1, coupling_lambda2, freq2, delta2, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma_1 = coupling_lambda1*freq1*F_function((E - mu - delta1*1j)/freq1)
        Sigma_2 = coupling_lambda2*freq2*F_function((E - mu - delta2*1j)/freq2)
        Sigma = Sigma_1 + Sigma_2 + 1j*gamma_array[E_num]/2

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array



def add_band_3_phonons_linear(k_array, E_array, spectrum, vF, kF, mu, coupling_lambda1, freq1, delta1, coupling_lambda2, freq2, delta2, coupling_lambda3, freq3, delta3, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma_1 = coupling_lambda1*freq1*F_function((E - mu - delta1*1j)/freq1)
        Sigma_2 = coupling_lambda2*freq2*F_function((E - mu - delta2*1j)/freq2)
        Sigma_3 = coupling_lambda3*freq3*F_function((E - mu - delta3*1j)/freq3)
        Sigma = Sigma_1 + Sigma_2 + Sigma_3

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array



def add_band_3_phonons_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda1, freq1, delta1, coupling_lambda2, freq2, delta2, coupling_lambda3, freq3, delta3, intensity):

    k_pnts = len(k_array); E_pnts = len(E_array)
    spectrum_band = np.zeros((E_pnts, k_pnts))
    ReS_array = np.zeros(len(E_array)); ReS_array[:] = np.nan
    ImS_array = np.zeros(len(E_array)); ImS_array[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        Sigma_1 = coupling_lambda1*freq1*F_function((E - mu - delta1*1j)/freq1)
        Sigma_2 = coupling_lambda2*freq2*F_function((E - mu - delta2*1j)/freq2)
        Sigma_3 = coupling_lambda3*freq3*F_function((E - mu - delta3*1j)/freq3)
        Sigma = Sigma_1 + Sigma_2 + Sigma_3 + 1j*gamma_array[E_num]/2

        for k_num in range(k_pnts):
            k = k_array[k_num]
            Ek = dispersion_linear(k, vF, kF, mu)
            G = intensity/((E - Ek) - Sigma)
            spectrum_band[E_num, k_num] = (1/pi)*G.imag
        
        if E < mu:
            ReS_array[E_num] = Sigma.real
            ImS_array[E_num] = Sigma.imag
    
    spectrum_new = spectrum + spectrum_band
    return spectrum_new, ReS_array, ImS_array





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Fermi Surfaces -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def cuprate_band(ky, mu_tb, t0, t1, t2, E, kx):
    f = - E - mu_tb - 2*t0*(np.cos(kx*pi) + np.cos(ky*pi)) - 4*t1*np.cos(kx*pi)*np.cos(ky*pi) - 2*t2*(np.cos(2*kx*pi) + np.cos(2*ky*pi))
    return f



def SRO_g_band(ky, mu_tb, t, tt, E, kx):
    f = - E + mu_tb - 2*t*(np.cos(kx*pi) + np.cos(ky*pi)) - 4*tt*np.cos(kx*pi)*np.cos(ky*pi)
    return f



def SRO_ab_band(ky, mu_tb, t, tt, sign, E, kx):
    f = - E + mu_tb - t*(np.cos(kx*pi) + np.cos(ky*pi)) + sign*np.sqrt((t**2)*((np.cos(kx*pi) - np.cos(ky*pi))**2) + 16*(tt**2)*((np.sin(kx*pi))**2)*((np.sin(ky*pi))**2))
    return f



def add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs, fwhm_k, intensity):

    conv_multiple = 6
    E_dist = conv_multiple*E_conv
    E_pnts = int(1 + 2*E_dist/dE)
    E_array = np.linspace(E0 - E_dist, E0 + E_dist, E_pnts, endpoint = True)
    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    spectrum_3D = np.zeros((E_pnts, ky_pnts, kx_pnts))

    mu_tb = coefs[0]; t0 = coefs[1]; t1 = coefs[2]; t2 = coefs[3]
    kx_pnts2 = int(np.floor((kx_pnts + 1)/2))
    kx_trace = np.zeros(4*kx_pnts2); ky_trace = np.zeros(4*kx_pnts2)
    kx_trace[:] = np.nan; ky_trace[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]

        for kx_num in range(kx_pnts):
            kx = kx_array[kx_num]

            if kx >= 0:
                kx_num2 = kx_num - (kx_pnts - kx_pnts2)
                kx_trace[kx_num2] = kx
                kx_trace[kx_num2 + kx_pnts2] = kx
                kx_trace[kx_num2 + 2*kx_pnts2] = -kx
                kx_trace[kx_num2 + 3*kx_pnts2] = -kx

                initial_guess = 0.5
                solution, info, ier, mesg = scipy.optimize.fsolve(cuprate_band, initial_guess, args = (mu_tb, t0, t1, t2, E, kx), full_output = True)

                if ier == 1:
                    # A solution is returned
                    ky_pos = np.absolute(solution)
                    ky_modulo = ky_pos%2

                    if 0 <= ky_modulo <= 1:
                        ky_peak = ky_modulo
                    else:
                        ky_peak = -(ky_modulo - 2)

                    ky_trace[kx_num2] = ky_peak
                    ky_trace[kx_num2 + kx_pnts2] = -ky_peak
                    ky_trace[kx_num2 + 2*kx_pnts2] = ky_peak
                    ky_trace[kx_num2 + 3*kx_pnts2] = -ky_peak

                    for ky_num in range(ky_pnts):
                        ky = ky_array[ky_num]

                        if ky >= 0:
                            spectral_intensity = (intensity/pi)*(fwhm_k/2)/((ky - ky_peak)**2 + (fwhm_k/2)**2)
                            spectrum_3D[E_num, ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, ky_num, -1 - kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, -1 - kx_num] = spectral_intensity
    
    dkx = (kx_array[-1] - kx_array[0])/(kx_pnts - 1)
    sigma_val_E = E_conv/dE
    sigma_val_kx = fwhm_k/dkx
    spectrum_convolve_E = scipy.ndimage.gaussian_filter(spectrum_3D, sigma = sigma_val_E, axes = 0, mode = 'nearest')
    map_cut = spectrum_convolve_E[int(np.round((E_pnts - 1)/2)),:,:]
    map_convolve_kx = scipy.ndimage.gaussian_filter(map_cut, sigma = sigma_val_kx/2.355, axes = 1, mode = 'nearest')
    map_band = map_raw + map_convolve_kx

    kx_single = kx_trace[:kx_pnts2]
    ky_single = ky_trace[:kx_pnts2]

    iter_start = 0
    while iter_start < kx_pnts2:
        if np.isnan(ky_single[iter_start]) == True:
            ky_single[iter_start] = 1
            iter_start += 1
        else:
            iter_start = kx_pnts2
    
    iter_end = 0
    while iter_end < kx_pnts2:
        if np.isnan(ky_single[kx_pnts2 - iter_end - 1]) == True:
            ky_single[kx_pnts2 - iter_end - 1] = 0
            iter_end += 1
        else:
            iter_end = kx_pnts2
    
    integral = scipy.integrate.trapezoid(ky_single, x = kx_single)
    doping = (0.5 - integral)*2
    return map_band, kx_trace, ky_trace, doping



def add_SRO_g_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs, fwhm_k, intensity):

    conv_multiple = 6
    E_dist = conv_multiple*E_conv
    E_pnts = int(1 + 2*E_dist/dE)
    E_array = np.linspace(E0 - E_dist, E0 + E_dist, E_pnts, endpoint = True)
    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    spectrum_3D = np.zeros((E_pnts, ky_pnts, kx_pnts))

    mu_tb = coefs[0]; t = coefs[1]; tt = coefs[2]
    kx_pnts2 = int(np.floor((kx_pnts + 1)/2))
    kx_trace = np.zeros(4*kx_pnts2); ky_trace = np.zeros(4*kx_pnts2)
    kx_trace[:] = np.nan; ky_trace[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]

        for kx_num in range(kx_pnts):
            kx = kx_array[kx_num]

            if kx >= 0:
                kx_num2 = kx_num - (kx_pnts - kx_pnts2)
                kx_trace[kx_num2] = kx
                kx_trace[kx_num2 + kx_pnts2] = kx
                kx_trace[kx_num2 + 2*kx_pnts2] = -kx
                kx_trace[kx_num2 + 3*kx_pnts2] = -kx

                initial_guess = 0.5
                solution, info, ier, mesg = scipy.optimize.fsolve(SRO_g_band, initial_guess, args = (mu_tb, t, tt, E, kx), full_output = True)

                if ier == 1:
                    # A solution is returned
                    ky_pos = np.absolute(solution)
                    ky_modulo = ky_pos%2

                    if 0 <= ky_modulo <= 1:
                        ky_peak = ky_modulo
                    else:
                        ky_peak = -(ky_modulo - 2)

                    ky_trace[kx_num2] = ky_peak
                    ky_trace[kx_num2 + kx_pnts2] = -ky_peak
                    ky_trace[kx_num2 + 2*kx_pnts2] = ky_peak
                    ky_trace[kx_num2 + 3*kx_pnts2] = -ky_peak

                    for ky_num in range(ky_pnts):
                        ky = ky_array[ky_num]

                        if ky >= 0:
                            spectral_intensity = (intensity/pi)*(fwhm_k/2)/((ky - ky_peak)**2 + (fwhm_k/2)**2)
                            spectrum_3D[E_num, ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, ky_num, -1 - kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, -1 - kx_num] = spectral_intensity
    
    dkx = (kx_array[-1] - kx_array[0])/(kx_pnts - 1)
    sigma_val_E = E_conv/dE
    sigma_val_kx = fwhm_k/dkx
    spectrum_convolve_E = scipy.ndimage.gaussian_filter(spectrum_3D, sigma = sigma_val_E, axes = 0, mode = 'nearest')
    map_cut = spectrum_convolve_E[int(np.round((E_pnts - 1)/2)),:,:]
    map_convolve_kx = scipy.ndimage.gaussian_filter(map_cut, sigma = sigma_val_kx/2.355, axes = 1, mode = 'nearest')
    map_band = map_raw + map_convolve_kx

    kx_single = kx_trace[:kx_pnts2]
    ky_single = ky_trace[:kx_pnts2]

    iter_start = 0
    while iter_start < kx_pnts2:
        if np.isnan(ky_single[iter_start]) == True:
            ky_single[iter_start] = 1
            iter_start += 1
        else:
            iter_start = kx_pnts2
    
    iter_end = 0
    while iter_end < kx_pnts2:
        if np.isnan(ky_single[kx_pnts2 - iter_end - 1]) == True:
            ky_single[kx_pnts2 - iter_end - 1] = 0
            iter_end += 1
        else:
            iter_end = kx_pnts2
    
    integral = scipy.integrate.trapezoid(ky_single, x = kx_single)
    doping = (0.5 - integral)*2
    return map_band, kx_trace, ky_trace, doping



def add_SRO_ab_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs, fwhm_k, intensity):

    conv_multiple = 6
    E_dist = conv_multiple*E_conv
    E_pnts = int(1 + 2*E_dist/dE)
    E_array = np.linspace(E0 - E_dist, E0 + E_dist, E_pnts, endpoint = True)
    kx_pnts = len(kx_array); ky_pnts = len(ky_array)
    spectrum_3D = np.zeros((E_pnts, ky_pnts, kx_pnts))

    mu_tb = coefs[0]; t = coefs[1]; tt = coefs[2]; sign = coefs[3]
    kx_pnts2 = int(np.floor((kx_pnts + 1)/2))
    kx_trace = np.zeros(4*kx_pnts2); ky_trace = np.zeros(4*kx_pnts2)
    kx_trace[:] = np.nan; ky_trace[:] = np.nan

    for E_num in range(E_pnts):
        E = E_array[E_num]
        
        for kx_num in range(kx_pnts):
            kx = kx_array[kx_num]

            if kx >= 0:
                kx_num2 = kx_num - (kx_pnts - kx_pnts2)
                kx_trace[kx_num2] = kx
                kx_trace[kx_num2 + kx_pnts2] = kx
                kx_trace[kx_num2 + 2*kx_pnts2] = -kx
                kx_trace[kx_num2 + 3*kx_pnts2] = -kx

                initial_guess = 0.5
                solution, info, ier, mesg = scipy.optimize.fsolve(SRO_ab_band, initial_guess, args = (mu_tb, t, tt, sign, E, kx), full_output = True)

                if ier == 1:
                    # A solution is returned
                    ky_pos = np.absolute(solution)
                    ky_modulo = ky_pos%2

                    if 0 <= ky_modulo <= 1:
                        ky_peak = ky_modulo
                    else:
                        ky_peak = -(ky_modulo - 2)

                    ky_trace[kx_num2] = ky_peak
                    ky_trace[kx_num2 + kx_pnts2] = -ky_peak
                    ky_trace[kx_num2 + 2*kx_pnts2] = ky_peak
                    ky_trace[kx_num2 + 3*kx_pnts2] = -ky_peak

                    for ky_num in range(ky_pnts):
                        ky = ky_array[ky_num]

                        if ky >= 0:
                            spectral_intensity = (intensity/pi)*(fwhm_k/2)/((ky - ky_peak)**2 + (fwhm_k/2)**2)
                            spectrum_3D[E_num, ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, kx_num] = spectral_intensity
                            spectrum_3D[E_num, ky_num, -1 - kx_num] = spectral_intensity
                            spectrum_3D[E_num, -1 - ky_num, -1 - kx_num] = spectral_intensity
    
    dkx = (kx_array[-1] - kx_array[0])/(kx_pnts - 1)
    sigma_val_E = E_conv/dE
    sigma_val_kx = fwhm_k/dkx
    spectrum_convolve_E = scipy.ndimage.gaussian_filter(spectrum_3D, sigma = sigma_val_E, axes = 0, mode = 'nearest')
    map_cut = spectrum_convolve_E[int(np.round((E_pnts - 1)/2)),:,:]
    map_convolve_kx = scipy.ndimage.gaussian_filter(map_cut, sigma = sigma_val_kx/2.355, axes = 1, mode = 'nearest')
    map_band = map_raw + map_convolve_kx

    kx_single = kx_trace[:kx_pnts2]
    ky_single = ky_trace[:kx_pnts2]

    iter_start = 0
    while iter_start < kx_pnts2:
        if np.isnan(ky_single[iter_start]) == True:
            ky_single[iter_start] = 1
            iter_start += 1
        else:
            iter_start = kx_pnts2
    
    iter_end = 0
    while iter_end < kx_pnts2:
        if np.isnan(ky_single[kx_pnts2 - iter_end - 1]) == True:
            ky_single[kx_pnts2 - iter_end - 1] = 0
            iter_end += 1
        else:
            iter_end = kx_pnts2
    
    integral = scipy.integrate.trapezoid(ky_single, x = kx_single)
    doping = (0.5 - integral)*2
    return map_band, kx_trace, ky_trace, doping





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Generate spectra/maps -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def spectrum_FD(spectrum_params, smooth_params, noise_params, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]

    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max, resolution, 1)
    spectrum_FD = multiply_FD(E_array, k_array, spectrum, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    spectrum_rescaled = spectrum_dynamic_range(spectrum_noise, dynamic_range)
    txt = convert_spectrum_to_txt(k_array, E_array, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array, spectrum_rescaled, full_name)
    write_to_text(txt, full_name, "_data")

    return mu, E_conv, txt



def spectrum_FL_linear(spectrum_params, smooth_params, noise_params, linear_band_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_band = add_band_qp_linear(k_array, E_array, spectrum, vF, kF, mu, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array/(np.absolute(vF)))
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    k_band_cut = cut_array(E_array, k_band, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    gamma_array_cut = cut_array(E_array, gamma_array, mu)/(np.absolute(vF))

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_fwhm_both(k_array, E_array_cut, k_band_cut, k_hwhm_left_cut, k_hwhm_right_cut, disp_array_cut, gamma_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E_array_cut, disp_array_cut, gamma_array_cut, txt



def spectrum_FL_quadratic(spectrum_params, smooth_params, noise_params, quadratic_band_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = quadratic_band_params[0]; kF = quadratic_band_params[1]; alpha = quadratic_band_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_quadratic(E_array, vF, kF, alpha, mu)
    spectrum_band = add_band_qp_quadratic(k_array, E_array, spectrum, vF, kF, alpha, mu, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    k_band_cut = cut_array(E_array, k_band, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    gamma_array_cut = cut_array(E_array, gamma_array, mu)

    k_hwhm_left_cut.fill(np.nan)
    k_hwhm_right_cut.fill(np.nan)
    gamma_array_cut.fill(np.nan)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_fwhm_both(k_array, E_array_cut, k_band_cut, k_hwhm_left_cut, k_hwhm_right_cut, disp_array_cut, gamma_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E_array_cut, disp_array_cut, txt



def spectrum_FL_superstructure_linear(spectrum_params, smooth_params, noise_params, linear_band_params, superstructure_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    superstructure_number = superstructure_params[0]; k_spacing = superstructure_params[1]; attenuation = superstructure_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_band = add_band_qp_linear(k_array, E_array, spectrum, vF, kF, mu, gamma_array, 1)

    iteration = 1
    while iteration < superstructure_number + 1:
        spectrum_band_2 = add_band_qp_linear(k_array, E_array, spectrum_band, vF, kF - iteration*k_spacing, mu, gamma_array, attenuation**iteration)
        spectrum_band = add_band_qp_linear(k_array, E_array, spectrum_band_2, vF, kF + iteration*k_spacing, mu, gamma_array, attenuation**iteration)
        iteration += 1

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array/(np.absolute(vF)))
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    k_band_cut = cut_array(E_array, k_band, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    gamma_array_cut = cut_array(E_array, gamma_array, mu)/(np.absolute(vF))

    k_hwhm_left_cut.fill(np.nan)
    k_hwhm_right_cut.fill(np.nan)
    gamma_array_cut.fill(np.nan)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_fwhm_both(k_array, E_array_cut, k_band_cut, k_hwhm_left_cut, k_hwhm_right_cut, disp_array_cut, gamma_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E_array_cut, disp_array_cut, txt



def spectrum_FL_band_bottom(spectrum_params, smooth_params, noise_params, band_bottom_params, gamma_intensity, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    E0 = band_bottom_params[0]; k0 = band_bottom_params[1]; alpha = band_bottom_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    Ek_band = get_Ek_band_bottom(k_array, band_bottom_params)
    gamma_array = create_gamma_FL(Ek_band, mu, gamma_intensity, T)
    spectrum_band = add_band_qp_band_bottom(k_array, E_array, spectrum, E0, k0, alpha, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    Ek_array_cut = cut_Ek_array(Ek_band, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_trace_k(k_array, E_array_cut, Ek_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return k_array, Ek_array_cut, txt



def spectrum_FL_Dirac(spectrum_params, smooth_params, noise_params, Dirac_cone_params, gamma_intensity, dynamic_range, plot_name):
    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    v0 = Dirac_cone_params[0]; k0 = Dirac_cone_params[1]; E0 = Dirac_cone_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)

    disp_array_1 = create_dispersion_linear(E_array, v0, k0, E0)
    spectrum_band_1 = add_band_qp_linear(k_array, E_array, spectrum, v0, k0, E0, gamma_array, 1)
    disp_array_2 = create_dispersion_linear(E_array, -v0, k0, E0)
    spectrum_band_2 = add_band_qp_linear(k_array, E_array, spectrum_band_1, -v0, k0, E0, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band_2, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    disp_array_cut_1 = cut_array(E_array, disp_array_1, mu)
    disp_array_cut_2 = cut_array(E_array, disp_array_2, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_trace_E_2(k_array, E_array_cut, disp_array_cut_1, disp_array_cut_2, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E0, txt



def spectrum_MFL_linear(spectrum_params, smooth_params, noise_params, linear_band_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_MFL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_band = add_band_qp_linear(k_array, E_array, spectrum, vF, kF, mu, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array/(np.absolute(vF)))
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    k_band_cut = cut_array(E_array, k_band, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    gamma_array_cut = cut_array(E_array, gamma_array, mu)/(np.absolute(vF))

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_fwhm_both(k_array, E_array_cut, k_band_cut, k_hwhm_left_cut, k_hwhm_right_cut, disp_array_cut, gamma_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E_array_cut, disp_array_cut, gamma_array_cut, txt



def spectrum_SC_linear(spectrum_params, smooth_params, noise_params, linear_band_params, BCS_linear_params, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    Gamma_0 = BCS_linear_params[0]; Gamma_1 = BCS_linear_params[1]; Delta = BCS_linear_params[2]

    E_min_shifted = E_min - mu
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min_shifted, -E_min_shifted, round(resolution/2), 0)
    spectrum_band = add_band_BCS_linear(k_array, E_array, spectrum, vF, kF, Gamma_0, Gamma_1, Delta, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)
    spectrum_symm = symmetrise(spectrum_noise)
    
    spectrum_rescaled = spectrum_dynamic_range(spectrum_symm, dynamic_range)

    txt = convert_spectrum_to_txt(k_array, E_array, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array, spectrum_rescaled, full_name)
    plot_spectrum_trace_gap(k_array, E_array, Delta, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return Delta, txt



def spectrum_SC_quadratic(spectrum_params, smooth_params, noise_params, quadratic_band_params, BCS_quadratic_params, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = quadratic_band_params[0]; kF = quadratic_band_params[1]; alpha = quadratic_band_params[2]
    Gamma_0 = BCS_quadratic_params[0]; Gamma_1 = BCS_quadratic_params[1]; Delta = BCS_quadratic_params[2]

    E_min_shifted = E_min - mu
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min_shifted, -E_min_shifted, round(resolution/2), 0)
    spectrum_band = add_band_BCS_quadratic(k_array, E_array, spectrum, vF, kF, alpha, Gamma_0, Gamma_1, Delta, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)
    spectrum_symm = symmetrise(spectrum_noise)
    
    spectrum_rescaled = spectrum_dynamic_range(spectrum_symm, dynamic_range)

    txt = convert_spectrum_to_txt(k_array, E_array, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array, spectrum_rescaled, full_name)
    plot_spectrum_trace_gap(k_array, E_array, Delta, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return Delta, txt



def spectrum_SC_band_bottom(spectrum_params, smooth_params, noise_params, band_bottom_params, BCS_band_bottom_params, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    E0 = band_bottom_params[0]; k0 = band_bottom_params[1]; alpha = band_bottom_params[2]
    Gamma_0 = BCS_band_bottom_params[0]; Gamma_1 = BCS_band_bottom_params[1]; Delta = BCS_band_bottom_params[2]

    E_min_shifted = E_min - mu
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min_shifted, -E_min_shifted, round(resolution/2), 0)
    spectrum_band = add_band_BCS_band_bottom(k_array, E_array, spectrum, E0 - mu, k0, alpha, Gamma_0, Gamma_1, Delta, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)
    spectrum_symm = symmetrise(spectrum_noise)
    
    spectrum_rescaled = spectrum_dynamic_range(spectrum_symm, dynamic_range)

    txt = convert_spectrum_to_txt(k_array, E_array, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array, spectrum_rescaled, full_name)
    plot_spectrum_trace_gap(k_array, E_array, Delta, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return Delta, txt



def spectrum_SC_2_band_bottoms(spectrum_params, smooth_params, noise_params, band_bottom_params_a, band_bottom_params_b, BCS_band_bottom_params_a, BCS_band_bottom_params_b, dynamic_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    E0_a = band_bottom_params_a[0]; k0_a = band_bottom_params_a[1]; alpha_a = band_bottom_params_a[2]
    E0_b = band_bottom_params_b[0]; k0_b = band_bottom_params_b[1]; alpha_b = band_bottom_params_b[2]
    Gamma_0_a = BCS_band_bottom_params_a[0]; Gamma_1_a = BCS_band_bottom_params_a[1]; Delta_a = BCS_band_bottom_params_a[2]
    Gamma_0_b = BCS_band_bottom_params_b[0]; Gamma_1_b = BCS_band_bottom_params_b[1]; Delta_b = BCS_band_bottom_params_b[2]

    E_min_shifted = E_min - mu
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min_shifted, -E_min_shifted, round(resolution/2), 0)
    spectrum_band_a = add_band_BCS_band_bottom(k_array, E_array, spectrum, E0_a - mu, k0_a, alpha_a, Gamma_0_a, Gamma_1_a, Delta_a, 1)
    spectrum_band_b = add_band_BCS_band_bottom(k_array, E_array, spectrum_band_a, E0_b - mu, k0_b, alpha_b, Gamma_0_b, Gamma_1_b, Delta_b, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band_b, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)
    spectrum_symm = symmetrise(spectrum_noise)
    
    spectrum_rescaled = spectrum_dynamic_range(spectrum_symm, dynamic_range)

    txt = convert_spectrum_to_txt(k_array, E_array, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array, spectrum_rescaled, full_name)
    plot_spectrum_trace_gap(k_array, E_array, Delta_a, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return Delta_a, txt



def spectrum_imp_linear(spectrum_params, smooth_params, noise_params, linear_band_params, fwhm, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    gamma_array = create_fwhm_constant(E_array, fwhm)
    #spectrum_band = add_band_qp_linear(k_array, E_array, spectrum, vF, kF, mu, gamma_array, 1)
    spectrum_band = add_band_MDC(k_array, E_array, vF, spectrum, disp_array, gamma_array, 1)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_band, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_band, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    k_band_cut = cut_array(E_array, k_band, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    gamma_array_cut = cut_array(E_array, gamma_array, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_fwhm_both(k_array, E_array_cut, k_band_cut, k_hwhm_left_cut, k_hwhm_right_cut, disp_array_cut, gamma_array_cut, spectrum_rescaled, full_name + "_trace")
    write_to_text(txt, full_name, "_data")

    return E_array_cut, disp_array_cut, gamma_array_cut, txt



def spectrum_phonon_1_FL(spectrum_params, smooth_params, noise_params, linear_band_params, phonon_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    coupling_lambda = phonon_params[0]; freq = phonon_params[1]; delta = phonon_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_phonon, ReS_array, ImS_array = add_band_phonon_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda, freq, delta, 1)
    ImS_array /= np.absolute(vF)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_phonon, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_phonon, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    k_phonon_cut = cut_array(E_array, k_phonon, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    ReS_array_cut = cut_array(E_array, ReS_array, mu)
    ImS_array_cut = cut_array(E_array, ImS_array, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); lambda_val = round(coupling_lambda, 2); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    lambda_str = str(lambda_val)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in lambda_str if ch not in exclude)
    append_parameters = f"_r{resolution}_n{noise_int}_l{s}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_1_phonon(k_array, E_array_cut, k_phonon_cut, disp_array_cut, k_hwhm_left_cut, k_hwhm_right_cut, mu - freq, spectrum_rescaled, full_name + "_trace")
    plot_sigma_1_phonon(E_array_cut, disp_array_cut, k_phonon_cut, k_hwhm_left_cut, k_hwhm_right_cut, ReS_array_cut, ImS_array_cut, mu, E_conv, freq, full_name)
    write_to_text(txt, full_name, "_data")

    return E_array_cut, mu - freq, disp_array_cut, k_phonon_cut, ReS_array_cut, ImS_array_cut, txt



def spectrum_phonon_1_MFL(spectrum_params, smooth_params, noise_params, linear_band_params, phonon_params, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    coupling_lambda = phonon_params[0]; freq = phonon_params[1]; delta = phonon_params[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_MFL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_phonon, ReS_array, ImS_array = add_band_phonon_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda, freq, delta, 1)
    ImS_array /= np.absolute(vF)

    spectrum_FD = multiply_FD(E_array, k_array, spectrum_phonon, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_phonon, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    k_phonon_cut = cut_array(E_array, k_phonon, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    ReS_array_cut = cut_array(E_array, ReS_array, mu)
    ImS_array_cut = cut_array(E_array, ImS_array, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); lambda_val = round(coupling_lambda, 2); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    lambda_str = str(lambda_val)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in lambda_str if ch not in exclude)
    append_parameters = f"_r{resolution}_n{noise_int}_l{s}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_1_phonon(k_array, E_array_cut, k_phonon_cut, disp_array_cut, k_hwhm_left_cut, k_hwhm_right_cut, mu - freq, spectrum_rescaled, full_name + "_trace")
    plot_sigma_1_phonon(E_array_cut, disp_array_cut, k_phonon_cut, k_hwhm_left_cut, k_hwhm_right_cut, ReS_array_cut, ImS_array_cut, mu, E_conv, freq, full_name)
    write_to_text(txt, full_name, "_data")

    return E_array_cut, mu - freq, disp_array_cut, k_phonon_cut, ReS_array_cut, ImS_array_cut, txt



def spectrum_phonon_2_FL(spectrum_params, smooth_params, noise_params, linear_band_params, phonon_params_1, phonon_params_2, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    coupling_lambda_1 = phonon_params_1[0]; freq_1 = phonon_params_1[1]; delta_1 = phonon_params_1[2]
    coupling_lambda_2 = phonon_params_2[0]; freq_2 = phonon_params_2[1]; delta_2 = phonon_params_2[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_phonon, ReS_array, ImS_array = add_band_2_phonons_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda_1, freq_1, delta_1, coupling_lambda_2, freq_2, delta_2, 1)
    ImS_array /= np.absolute(vF)
    
    spectrum_FD = multiply_FD(E_array, k_array, spectrum_phonon, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_phonon, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    k_phonon_cut = cut_array(E_array, k_phonon, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    ReS_array_cut = cut_array(E_array, ReS_array, mu)
    ImS_array_cut = cut_array(E_array, ImS_array, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_2_phonons(k_array, E_array_cut, k_phonon_cut, disp_array_cut, k_hwhm_left_cut, k_hwhm_right_cut, mu - freq_1, mu - freq_2, spectrum_rescaled, full_name + "_trace")
    plot_sigma_2_phonons(E_array_cut, disp_array_cut, k_phonon_cut, k_hwhm_left_cut, k_hwhm_right_cut, ReS_array_cut, ImS_array_cut, mu, E_conv, freq_1, freq_2, full_name)
    write_to_text(txt, full_name, "_data")

    return E_array_cut, mu - freq_1, mu - freq_2, disp_array_cut, k_phonon_cut, ReS_array_cut, ImS_array_cut, txt



def spectrum_phonon_3_FL(spectrum_params, smooth_params, noise_params, linear_band_params, phonon_params_1, phonon_params_2, phonon_params_3, gamma_intensity, dynamic_range, k_int_range, plot_name):

    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]
    T = smooth_params[0]; k_conv = smooth_params[1]; E_conv = smooth_params[2]
    noise_pnts = round(noise_params[0]); sigma_E = noise_params[1]; sigma_k = noise_params[2]; noise_ratio = noise_params[3]
    vF = linear_band_params[0]; kF = linear_band_params[1]
    coupling_lambda_1 = phonon_params_1[0]; freq_1 = phonon_params_1[1]; delta_1 = phonon_params_1[2]
    coupling_lambda_2 = phonon_params_2[0]; freq_2 = phonon_params_2[1]; delta_2 = phonon_params_2[2]
    coupling_lambda_3 = phonon_params_3[0]; freq_3 = phonon_params_3[1]; delta_3 = phonon_params_3[2]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    gamma_array = create_gamma_FL(E_array, mu, gamma_intensity, T)
    disp_array = create_dispersion_linear(E_array, vF, kF, mu)
    spectrum_phonon, ReS_array, ImS_array = add_band_3_phonons_linear_interaction(k_array, E_array, gamma_array, spectrum, vF, kF, mu, coupling_lambda_1, freq_1, delta_1, coupling_lambda_2, freq_2, delta_2, coupling_lambda_3, freq_3, delta_3, 1)
    ImS_array /= np.absolute(vF)
    
    spectrum_FD = multiply_FD(E_array, k_array, spectrum_phonon, mu, T)
    spectrum_conv_k = convolve_k(k_array, E_array, spectrum_FD, k_conv)
    spectrum_conv_E = convolve_E(k_array, E_array, spectrum_conv_k, E_conv)
    spectrum_noise = add_noise_scaled_spectrum(k_array, E_array, spectrum_conv_E, noise_pnts, sigma_E, sigma_k, mu, T, E_conv, noise_ratio)

    gamma_array_trunc, k_phonon, k_hwhm_left, k_hwhm_right = get_fwhm_array(E_array, k_array, mu, spectrum_noise, k_int_range, gamma_array)
    spectrum_cut = cut_spectrum(E_array, spectrum_noise, mu)
    spectrum_rescaled = spectrum_dynamic_range(spectrum_cut, dynamic_range)

    E_array_cut = cut_array(E_array, E_array, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    k_phonon_cut = cut_array(E_array, k_phonon, mu)
    k_hwhm_left_cut = cut_array(E_array, k_hwhm_left, mu)
    k_hwhm_right_cut = cut_array(E_array, k_hwhm_right, mu)
    disp_array_cut = cut_array(E_array, disp_array, mu)
    ReS_array_cut = cut_array(E_array, ReS_array, mu)
    ImS_array_cut = cut_array(E_array, ImS_array, mu)

    txt = convert_spectrum_to_txt(k_array, E_array_cut, spectrum_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio); k_int = round(1000*k_conv); E_int = round(1000*E_conv)
    append_parameters = f"_r{resolution}_n{noise_int}_k{k_int}_e{E_int}"
    full_name = plot_name + append_parameters

    plot_spectrum(k_array, E_array_cut, spectrum_rescaled, full_name)
    plot_spectrum_3_phonons(k_array, E_array_cut, k_phonon_cut, disp_array_cut, k_hwhm_left_cut, k_hwhm_right_cut, mu - freq_1, mu - freq_2,  mu - freq_3, spectrum_rescaled, full_name + "_trace")
    plot_sigma_3_phonons(E_array_cut, disp_array_cut, k_phonon_cut, k_hwhm_left_cut, k_hwhm_right_cut, ReS_array_cut, ImS_array_cut, mu, E_conv, freq_1, freq_2, freq_3, full_name)
    write_to_text(txt, full_name, "_data")

    return E_array_cut, mu - freq_1, mu - freq_2, mu - freq_3, disp_array_cut, k_phonon_cut, ReS_array_cut, ImS_array_cut, txt



def map_cuprate_monolayer_band(map_params, E_params, noise_params, k_conv, coefs, dynamic_range, plot_name):

    resolution = map_params[0]; xy_bound = map_params[1]; bkg = map_params[2]
    E0 = E_params[0]; E_conv = E_params[1]; dE = E_params[2]
    noise_pnts = round(noise_params[0]); sigma_k = noise_params[1]; noise_ratio = noise_params[2]

    kx_array, ky_array, map_raw = create_map(-xy_bound, xy_bound, resolution, -xy_bound, xy_bound, resolution, bkg)
    map_band, kx_trace, ky_trace, doping = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs, k_conv, 1)
    map_total = map_band + np.transpose(map_band)

    map_noise = add_noise_scaled_map(kx_array, ky_array, map_total, noise_pnts, sigma_k, noise_ratio)
    map_rescaled = spectrum_dynamic_range(map_noise, dynamic_range)

    txt = convert_map_to_txt(kx_array, ky_array, map_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{resolution}_n{noise_int}"
    full_name = plot_name + append_parameters

    plot_map_trace(kx_array, ky_array, kx_trace, ky_trace, map_rescaled, full_name + "_trace")
    plot_map(kx_array, ky_array, map_rescaled, full_name)
    write_to_text(txt, full_name, "_data")

    return doping, txt



def map_cuprate_bilayer_bands(map_params, E_params, noise_params, k_conv, coefs_1, coefs_2, dynamic_range, plot_name):
    
    resolution = map_params[0]; xy_bound = map_params[1]; bkg = map_params[2]
    E0 = E_params[0]; E_conv = E_params[1]; dE = E_params[2]
    noise_pnts = round(noise_params[0]); sigma_k = noise_params[1]; noise_ratio = noise_params[2]

    kx_array, ky_array, map_raw = create_map(-xy_bound, xy_bound, resolution, -xy_bound, xy_bound, resolution, bkg)
    map_band_1, kx_trace_1, ky_trace_1, doping_1 = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs_1, k_conv, 1)
    map_band_2, kx_trace_2, ky_trace_2, doping_2 = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_band_1, coefs_2, k_conv, 1)
    map_total = map_band_2 + np.transpose(map_band_2)

    map_noise = add_noise_scaled_map(kx_array, ky_array, map_total, noise_pnts, sigma_k, noise_ratio)
    map_rescaled = spectrum_dynamic_range(map_noise, dynamic_range)

    txt = convert_map_to_txt(kx_array, ky_array, map_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{resolution}_n{noise_int}"
    full_name = plot_name + append_parameters

    kx_trace = np.concatenate((kx_trace_1, kx_trace_2))
    ky_trace = np.concatenate((ky_trace_1, ky_trace_2))

    plot_map_trace(kx_array, ky_array, kx_trace, ky_trace, map_rescaled, full_name + "_trace")
    plot_map(kx_array, ky_array, map_rescaled, full_name)
    write_to_text(txt, full_name, "_data")

    return doping_1, doping_2, txt



def map_SRO_bands(map_params, E_params, noise_params, k_conv, coefs_1, coefs_2, coefs_3, dynamic_range, plot_name):
    
    resolution = map_params[0]; xy_bound = map_params[1]; bkg = map_params[2]
    E0 = E_params[0]; E_conv = E_params[1]; dE = E_params[2]
    noise_pnts = round(noise_params[0]); sigma_k = noise_params[1]; noise_ratio = noise_params[2]

    kx_array, ky_array, map_raw = create_map(-xy_bound, xy_bound, resolution, -xy_bound, xy_bound, resolution, bkg)
    map_band_1, kx_trace_1, ky_trace_1, doping_1 = add_SRO_ab_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs_1, k_conv, 1)
    map_band_2, kx_trace_2, ky_trace_2, doping_2 = add_SRO_ab_band(E0, E_conv, dE, kx_array, ky_array, map_band_1, coefs_2, k_conv, 1)
    map_band_3, kx_trace_3, ky_trace_3, doping_3 = add_SRO_g_band(E0, E_conv, dE, kx_array, ky_array, map_band_2, coefs_3, k_conv, 1)
    map_total = map_band_3 + np.transpose(map_band_3)

    map_noise = add_noise_scaled_map(kx_array, ky_array, map_total, noise_pnts, sigma_k, noise_ratio)
    map_rescaled = spectrum_dynamic_range(map_noise, dynamic_range)

    txt = convert_map_to_txt(kx_array, ky_array, map_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{resolution}_n{noise_int}"
    full_name = plot_name + append_parameters

    kx_trace = np.concatenate((kx_trace_1, kx_trace_2, kx_trace_3))
    ky_trace = np.concatenate((ky_trace_1, ky_trace_2, ky_trace_3))

    plot_map_trace(kx_array, ky_array, kx_trace, ky_trace, map_rescaled, full_name + "_trace")
    plot_map(kx_array, ky_array, map_rescaled, full_name)
    write_to_text(txt, full_name, "_data")

    return doping_1, doping_2, doping_3, txt



def map_nickelate_trilayer_bands(map_params, E_params, noise_params, k_conv, coefs_1, coefs_2, coefs_3, dynamic_range, plot_name):
    
    resolution = map_params[0]; xy_bound = map_params[1]; bkg = map_params[2]
    E0 = E_params[0]; E_conv = E_params[1]; dE = E_params[2]
    noise_pnts = round(noise_params[0]); sigma_k = noise_params[1]; noise_ratio = noise_params[2]

    kx_array, ky_array, map_raw = create_map(-xy_bound, xy_bound, resolution, -xy_bound, xy_bound, resolution, bkg)
    map_band_1, kx_trace_1, ky_trace_1, doping_1 = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_raw, coefs_1, k_conv, 1)
    map_band_2, kx_trace_2, ky_trace_2, doping_2 = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_band_1, coefs_2, k_conv, 1)
    map_band_3, kx_trace_3, ky_trace_3, doping_3 = add_cuprate_band(E0, E_conv, dE, kx_array, ky_array, map_band_2, coefs_3, k_conv, 1)
    map_total = map_band_3 + np.transpose(map_band_3)

    map_noise = add_noise_scaled_map(kx_array, ky_array, map_total, noise_pnts, sigma_k, noise_ratio)
    map_rescaled = spectrum_dynamic_range(map_noise, dynamic_range)

    txt = convert_map_to_txt(kx_array, ky_array, map_rescaled)
    tokens = token_counter(txt)
    print("Estimated token count:", tokens)

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{resolution}_n{noise_int}"
    full_name = plot_name + append_parameters

    kx_trace = np.concatenate((kx_trace_1, kx_trace_2, kx_trace_3))
    ky_trace = np.concatenate((ky_trace_1, ky_trace_2, ky_trace_3))

    plot_map_trace(kx_array, ky_array, kx_trace, ky_trace, map_rescaled, full_name + "_trace")
    plot_map(kx_array, ky_array, map_rescaled, full_name)
    write_to_text(txt, full_name, "_data")

    return doping_1, doping_2, doping_3, txt





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------- Commonly used parameters for examples ------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_default_noise():
    noise_pnts = 20000; sigma_E = 0.005; sigma_k = 0.01
    noise_array = np.array([noise_pnts, sigma_E, sigma_k])
    return noise_array



def get_default_map_noise():
    noise_pnts = 20000; sigma_k = 0.01
    noise_array = np.array([noise_pnts, sigma_k])
    return noise_array



def get_spectrum_params_a():
    k_min = -1; k_max = 1; mu = 2.71; E_min = 2.5; E_max = 2.8
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_spectrum_params_extended_a():
    k_min = -1; k_max = 1; mu = 2.71; E_min = mu - 0.2; E_max = 2.8
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_band_linear_params_a():
    vF = 1.05; kF = 0.4
    linear_band_params = np.array([vF, kF])
    k_int_range = np.array([0.75, 1])
    return linear_band_params, k_int_range

def get_band_quadratic_params_a():
    vF = 0.11; kF = 0.7; alpha = -0.2
    quadratic_band_params = np.array([vF, kF, alpha])
    k_int_range = np.array([-1, -0.75])
    return quadratic_band_params, k_int_range

def get_band_bottom_params_1_a():
    E0 = 2.55; k0 = 0; alpha = 1
    band_bottom_params = np.array([E0, k0, alpha])
    return band_bottom_params

def get_band_bottom_params_2_a():
    E0 = 2.67; k0 = 0; alpha = 1
    band_bottom_params = np.array([E0, k0, alpha])
    return band_bottom_params

def get_Dirac_cone_params_a():
    v0 = 0.31; k0 = 0; E0 = 2.65
    linear_band_params = np.array([v0, k0, E0])
    return linear_band_params

def get_1_phonon_a():
    freq = 0.072; delta = 0.001
    phonon_params = np.array([freq, delta])
    return phonon_params

def get_2_phonons_a():
    freq_1 = 0.072; delta_1 = 0.001
    freq_2 = 0.099; delta_2 = 0.005
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    return phonon_params_1, phonon_params_2

def get_3_phonons_a():
    freq_1 = 0.016; delta_1 = 0.003
    freq_2 = 0.072; delta_2 = 0.001
    freq_3 = 0.125; delta_3 = 0.005
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    phonon_params_3 = np.array([freq_3, delta_3])
    return phonon_params_1, phonon_params_2, phonon_params_3



def get_spectrum_params_b():
    k_min = -2.4; k_max = -1.8; mu = 15.98; E_min = 15.75; E_max = 16.05
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_spectrum_params_extended_b():
    k_min = -2.4; k_max = -1.8; mu = 15.98; E_min = mu - 0.2; E_max = 16.05
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_band_linear_params_b():
    vF = -1.4; kF = -2.3
    linear_band_params = np.array([vF, kF])
    k_int_range = np.array([-2.4, -2.25])
    return linear_band_params, k_int_range

def get_band_quadratic_params_b():
    vF = -2; kF = -2.3; alpha = 2.5
    quadratic_band_params = np.array([vF, kF, alpha])
    k_int_range = np.array([-2.4, -2.25])
    return quadratic_band_params, k_int_range

def get_band_bottom_params_1_b():
    E0 = 15.8; k0 = -2; alpha = 2
    band_bottom_params = np.array([E0, k0, alpha])
    return band_bottom_params

def get_Dirac_cone_params_b():
    v0 = 2; k0 = -2; E0 = 15.9
    linear_band_params = np.array([v0, k0, E0])
    return linear_band_params

def get_1_phonon_b():
    freq = 0.03; delta = 0.002
    phonon_params = np.array([freq, delta])
    return phonon_params

def get_2_phonons_b():
    freq_1 = 0.024; delta_1 = 0.001
    freq_2 = 0.087; delta_2 = 0.0027
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    return phonon_params_1, phonon_params_2

def get_3_phonons_b():
    freq_1 = 0.026; delta_1 = 0.0009
    freq_2 = 0.051; delta_2 = 0.002
    freq_3 = 0.093; delta_3 = 0.004
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    phonon_params_3 = np.array([freq_3, delta_3])
    return phonon_params_1, phonon_params_2, phonon_params_3



def get_spectrum_params_c():
    k_min = 3.2; k_max = 4.0; mu = 8.01; E_min = 7.72; E_max = 8.15
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_spectrum_params_extended_c():
    k_min = 3.2; k_max = 4.0; mu = 8.01; E_min = mu - 0.2; E_max = 8.15
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_band_linear_params_c():
    vF = 1.8; kF = 3.8
    linear_band_params = np.array([vF, kF])
    k_int_range = np.array([3.9, 4.0])
    return linear_band_params, k_int_range

def get_band_quadratic_params_c():
    vF = 1.8; kF = 3.8; alpha = 2
    quadratic_band_params = np.array([vF, kF, alpha])
    k_int_range = np.array([3.9, 4.0])
    return quadratic_band_params, k_int_range

def get_band_bottom_params_1_c():
    E0 = 7.8; k0 = 3.7; alpha = 2
    band_bottom_params = np.array([E0, k0, alpha])
    return band_bottom_params

def get_Dirac_cone_params_c():
    v0 = 2.2; k0 = 3.5; E0 = 7.88
    linear_band_params = np.array([v0, k0, E0])
    return linear_band_params

def get_1_phonon_c():
    freq = 0.02; delta = 0.006
    phonon_params = np.array([freq, delta])
    return phonon_params

def get_2_phonons_c():
    freq_1 = 0.01; delta_1 = 0.0014
    freq_2 = 0.117; delta_2 = 0.0037
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    return phonon_params_1, phonon_params_2

def get_3_phonons_c():
    freq_1 = 0.008; delta_1 = 0.0012
    freq_2 = 0.043; delta_2 = 0.0025
    freq_3 = 0.106; delta_3 = 0.0048
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    phonon_params_3 = np.array([freq_3, delta_3])
    return phonon_params_1, phonon_params_2, phonon_params_3



def get_spectrum_params_d():
    k_min = 1.2; k_max = 3.2; mu = 24.98; E_min = 24.83; E_max = 25.05
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_spectrum_params_extended_d():
    k_min = 1.2; k_max = 3.2; mu = 24.98; E_min = mu - 0.2; E_max = 25.05
    params_array = np.array([k_min, k_max, mu, E_min, E_max])
    return params_array

def get_band_linear_params_d():
    vF = -0.25; kF = 1.4
    linear_band_params = np.array([vF, kF])
    k_int_range = np.array([1.2, 1.25])
    return linear_band_params, k_int_range

def get_band_quadratic_params_d():
    vF = -0.05; kF = 1.4; alpha = -0.07
    quadratic_band_params = np.array([vF, kF, alpha])
    k_int_range = np.array([1.2, 1.25])
    return quadratic_band_params, k_int_range

def get_band_bottom_params_1_d():
    E0 = 24.9; k0 = 2; alpha = 0.5
    band_bottom_params = np.array([E0, k0, alpha])
    return band_bottom_params

def get_Dirac_cone_params_d():
    v0 = 0.2; k0 = 1.9; E0 = 24.97
    linear_band_params = np.array([v0, k0, E0])
    return linear_band_params

def get_1_phonon_d():
    freq = 0.051; delta = 0.0006
    phonon_params = np.array([freq, delta])
    return phonon_params

def get_2_phonons_d():
    freq_1 = 0.033; delta_1 = 0.001
    freq_2 = 0.052; delta_2 = 0.0033
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    return phonon_params_1, phonon_params_2

def get_3_phonons_d():
    freq_1 = 0.032; delta_1 = 0.0008
    freq_2 = 0.057; delta_2 = 0.0022
    freq_3 = 0.114; delta_3 = 0.004
    phonon_params_1 = np.array([freq_1, delta_1])
    phonon_params_2 = np.array([freq_2, delta_2])
    phonon_params_3 = np.array([freq_3, delta_3])
    return phonon_params_1, phonon_params_2, phonon_params_3



def get_BCS_params_a():
    Gamma_0 = 0.001; Gamma_1 = 0.02; Delta = 0.032
    BCS_params = np.array([Gamma_0, Gamma_1, Delta])
    return BCS_params

def get_BCS_params_b():
    Gamma_0 = 0.002; Gamma_1 = 0.05; Delta = 0.017
    BCS_params = np.array([Gamma_0, Gamma_1, Delta])
    return BCS_params

def get_BCS_params_c():
    Gamma_0 = 0.003; Gamma_1 = 0.03; Delta = 0.025
    BCS_params = np.array([Gamma_0, Gamma_1, Delta])
    return BCS_params

def get_BCS_params_d():
    Gamma_0 = 0.002; Gamma_1 = 0.02; Delta = 0.008
    BCS_params = np.array([Gamma_0, Gamma_1, Delta])
    return BCS_params



def get_cuprate_monolayer_coefs_a():
    mu = 0.25; t0 = -0.166; t1 = 0.0782; t2 = -0.001
    coefs = np.array([mu, t0, t1, t2])
    return coefs

def get_cuprate_monolayer_coefs_b():
    mu = 0.27; t0 = -0.166; t1 = 0.0782; t2 = -0.001
    coefs = np.array([mu, t0, t1, t2])
    return coefs

def get_cuprate_monolayer_coefs_c():
    mu = 0.33; t0 = -0.166; t1 = 0.0782; t2 = -0.001
    coefs = np.array([mu, t0, t1, t2])
    return coefs

def get_cuprate_monolayer_coefs_d():
    mu = 0.32; t0 = -0.166; t1 = 0.0782; t2 = -0.001
    coefs = np.array([mu, t0, t1, t2])
    return coefs



def get_cuprate_bilayer_coefs_a():
    mu_1 = 0.25; t0_1 = -0.166; t1_1 = 0.0782; t2_1 = -0.001
    mu_2 = 0.24; t0_2 = -0.166; t1_2 = 0.09; t2_2 = -0.001
    coefs_1 = np.array([mu_1, t0_1, t1_1, t2_1])
    coefs_2 = np.array([mu_2, t0_2, t1_2, t2_2])
    return coefs_1, coefs_2

def get_cuprate_bilayer_coefs_b():
    mu_1 = 0.32; t0_1 = -0.166; t1_1 = 0.0782; t2_1 = -0.001
    mu_2 = 0.31; t0_2 = -0.166; t1_2 = 0.082; t2_2 = -0.001
    coefs_1 = np.array([mu_1, t0_1, t1_1, t2_1])
    coefs_2 = np.array([mu_2, t0_2, t1_2, t2_2])
    return coefs_1, coefs_2



def get_SRO_coefs_a():
    mu_a = -0.3; t_a = 0.25; tt_a = 0.0375
    mu_g = -0.55; t_g = 0.35; tt_g = 0.15
    coefs_a = np.array([mu_a, t_a, tt_a, 1])
    coefs_b = np.array([mu_a, t_a, tt_a, -1])
    coefs_g = np.array([mu_g, t_g, tt_g])
    return coefs_a, coefs_b, coefs_g

def get_SRO_coefs_b():
    mu_a = -0.25; t_a = 0.25; tt_a = 0.0375
    mu_g = -0.65; t_g = 0.35; tt_g = 0.15
    coefs_a = np.array([mu_a, t_a, tt_a, 1])
    coefs_b = np.array([mu_a, t_a, tt_a, -1])
    coefs_g = np.array([mu_g, t_g, tt_g])
    return coefs_a, coefs_b, coefs_g



def get_nickelate_trilayer_coefs_a():
    t0_1 = 410.4E-3; mu_1 = -1.384*t0_1; t1_1 = -0.1532*t0_1; t2_1 = 0.0719*t0_1
    t0_2 = 426.2E-3; mu_2 = -1.037*t0_2; t1_2 = -0.2505*t0_2; t2_2 = 0.1071*t0_2
    t0_3 = 422.1E-3; mu_3 = -1.138*t0_3; t1_3 = -0.2205*t0_3; t2_3 = 0.0988*t0_3
    coefs_1 = np.array([mu_1, t0_1, t1_1, t2_1])
    coefs_2 = np.array([mu_2, t0_2, t1_2, t2_2])
    coefs_3 = np.array([mu_3, t0_3, t1_3, t2_3])
    return coefs_1, coefs_2, coefs_3

def get_nickelate_trilayer_coefs_b():
    t0_1 = 410.4E-3; mu_1 = -1.184*t0_1; t1_1 = -0.1532*t0_1; t2_1 = 0.0719*t0_1
    t0_2 = 426.2E-3; mu_2 = -0.837*t0_2; t1_2 = -0.2505*t0_2; t2_2 = 0.1071*t0_2
    t0_3 = 422.1E-3; mu_3 = -0.938*t0_3; t1_3 = -0.2205*t0_3; t2_3 = 0.0988*t0_3
    coefs_1 = np.array([mu_1, t0_1, t1_1, t2_1])
    coefs_2 = np.array([mu_2, t0_2, t1_2, t2_2])
    coefs_3 = np.array([mu_3, t0_3, t1_3, t2_3])
    return coefs_1, coefs_2, coefs_3





# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------ A1 ----- Fermi Level extraction ------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_A1(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000
    
    T_a = 300
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    plot_name_a = "Plots/A1/A1a"

    T_b = 20
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    plot_name_b = "Plots/A1/A1b"

    T_c = 150
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    plot_name_c = "Plots/A1/A1c"

    T_d = 73
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    plot_name_d = "Plots/A1/A1d"

    mu_a, E_conv_a, txt_a = spectrum_FD(spectrum_params_a, smooth_params_a, noise_params_a, dynamic_range, plot_name_a)
    mu_b, E_conv_b, txt_b = spectrum_FD(spectrum_params_b, smooth_params_b, noise_params_b, dynamic_range, plot_name_b)
    mu_c, E_conv_c, txt_c = spectrum_FD(spectrum_params_c, smooth_params_c, noise_params_c, dynamic_range, plot_name_c)
    mu_d, E_conv_d, txt_d = spectrum_FD(spectrum_params_d, smooth_params_d, noise_params_d, dynamic_range, plot_name_d)

    return mu_a, mu_b, mu_c, mu_d, txt_a, txt_b, txt_c, txt_d



def ask_A1(resolution, size, noise_ratio):
    
    mu_a, mu_b, mu_c, mu_d, txt_a, txt_b, txt_c, txt_d = get_A1(int(resolution), noise_ratio)
    mu_a = np.round(mu_a, decimals = 4); mu_b = np.round(mu_b, decimals = 4); mu_c = np.round(mu_c, decimals = 4); mu_d = np.round(mu_d, decimals = 4)
    num = 0

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        f'Read "Dataset A". The Fermi energy of "Dataset A" is {mu_a} eV. '\
        f'Read "Dataset B". The Fermi energy of "Dataset B" is {mu_b} eV. '\
        f'Read "Dataset C". The Fermi energy of "Dataset C" is {mu_c} eV. '\
        'Now read "Dataset D". State the Fermi energy of "Dataset D" in units of electron-Volts. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/A1/A1"
    elif size == 2:
        question_name = "Prompts_med/A1/A1"
    elif size == 3:
        question_name = "Prompts_large/A1/A1"
    else:
        question_name = "Prompts_single/A1/A1"
    
    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(mu_d), full_name, "_S")
    else:
        write_to_text(str(mu_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------- B1 ----- Linear dispersion --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B1(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    vF_a = linear_band_params_a[0]
    plot_name_a = "Plots/B1/B1a"

    T_b = 25
    gamma_intensity_b = 2
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    vF_b = linear_band_params_b[0]
    plot_name_b = "Plots/B1/B1b"

    T_c = 50
    gamma_intensity_c = 1.5
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    vF_c = linear_band_params_c[0]
    plot_name_c = "Plots/B1/B1c"

    T_d = 220
    gamma_intensity_d = 1.2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    vF_d = linear_band_params_d[0]
    plot_name_d = "Plots/B1/B1d"

    E_array_cut_a, disp_array_cut_a, gamma_array_cut_a, txt_a = spectrum_FL_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, gamma_array_cut_b, txt_b = spectrum_FL_linear(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, gamma_array_cut_c, txt_c = spectrum_FL_linear(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, gamma_array_cut_d, txt_d = spectrum_FL_linear(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d



def ask_B1_dispersion(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B1(int(resolution), noise_ratio)
    disp_array_cut_a = np.round(disp_array_cut_a, decimals = 4); disp_array_cut_b = np.round(disp_array_cut_b, decimals = 4); disp_array_cut_c = np.round(disp_array_cut_c, decimals = 4); disp_array_cut_d = np.round(disp_array_cut_d, decimals = 4)
    num = len(disp_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. Here are three examples with linear dispersions. '\
        'Read "Dataset A". The linear dispersion of "Dataset A" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_a) + ']. '\
        'Read "Dataset B". The linear dispersion of "Dataset B" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_b) + ']. '\
        'Read "Dataset C". The linear dispersion of "Dataset C" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the linear dispersion of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B1/B1"
    elif size == 2:
        question_name = "Prompts_med/B1/B1"
    elif size == 3:
        question_name = "Prompts_large/B1/B1"
    else:
        question_name = "Prompts_single/B1/B1"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(disp_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(disp_array_cut_d), question_name, "_S")

    return question, num



def ask_B1_vF(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B1(int(resolution), noise_ratio)
    vF_a = np.round(vF_a, decimals = 4); vF_b = np.round(vF_b, decimals = 4); vF_c = np.round(vF_c, decimals = 4); vF_d = np.round(vF_d, decimals = 4)
    num = 0

    prompt = 'Four datasets containing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Fermi velocity is the gradient of a dispersion at the Fermi energy. Here are three examples with linear dispersions. '\
        f'Read "Dataset A". The Fermi velocity of "Dataset A" is {vF_a}. '\
        f'Read "Dataset B". The Fermi velocity of "Dataset B" is {vF_b}. '\
        f'Read "Dataset C". The Fermi velocity of "Dataset C" is {vF_c}. '\
        'Now read "Dataset D", which has a linear dispersion. State the Fermi velocity of "Dataset D" as an single number. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B1/B1_vF"
    elif size == 2:
        question_name = "Prompts_med/B1/B1_vF"
    elif size == 3:
        question_name = "Prompts_large/B1/B1_vF"
    else:
        question_name = "Prompts_single/B1/B1_vF"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(vF_d), full_name, "_S")
    else:
        write_to_text(str(vF_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------ B2 ----- Quadratic dispersion ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B2(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    quadratic_band_params_a, k_int_range_a = get_band_quadratic_params_a()
    vF_a = quadratic_band_params_a[0]
    plot_name_a = "Plots/B2/B2a"

    T_b = 30
    gamma_intensity_b = 1
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    quadratic_band_params_b, k_int_range_b = get_band_quadratic_params_b()
    vF_b = quadratic_band_params_b[0]
    plot_name_b = "Plots/B2/B2b"

    T_c = 100
    gamma_intensity_c = 1
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    quadratic_band_params_c, k_int_range_c = get_band_quadratic_params_c()
    vF_c = quadratic_band_params_c[0]
    plot_name_c = "Plots/B2/B2c"

    T_d = 125
    gamma_intensity_d = 1
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    quadratic_band_params_d, k_int_range_d = get_band_quadratic_params_d()
    vF_d = quadratic_band_params_d[0]
    plot_name_d = "Plots/B2/B2d"

    E_array_cut_a, disp_array_cut_a, txt_a = spectrum_FL_quadratic(spectrum_params_a, smooth_params_a, noise_params_a, quadratic_band_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, txt_b = spectrum_FL_quadratic(spectrum_params_b, smooth_params_b, noise_params_b, quadratic_band_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, txt_c = spectrum_FL_quadratic(spectrum_params_c, smooth_params_c, noise_params_c, quadratic_band_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, txt_d = spectrum_FL_quadratic(spectrum_params_d, smooth_params_d, noise_params_d, quadratic_band_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d



def ask_B2_dispersion(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B2(int(resolution), noise_ratio)
    disp_array_cut_a = np.round(disp_array_cut_a, decimals = 4); disp_array_cut_b = np.round(disp_array_cut_b, decimals = 4); disp_array_cut_c = np.round(disp_array_cut_c, decimals = 4); disp_array_cut_d = np.round(disp_array_cut_d, decimals = 4)
    num = len(disp_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. Here are three examples with quadratic dispersions. '\
        'Read "Dataset A". The quadratic dispersion of "Dataset A" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_a) + ']. '\
        'Read "Dataset B". The quadratic dispersion of "Dataset B" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_b) + ']. '\
        'Read "Dataset C". The quadratic dispersion of "Dataset C" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the quadratic dispersion of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B2/B2"
    elif size == 2:
        question_name = "Prompts_med/B2/B2"
    elif size == 3:
        question_name = "Prompts_large/B2/B2"
    else:
        question_name = "Prompts_single/B2/B2"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(disp_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(disp_array_cut_d), question_name, "_S")

    return question, num



def ask_B2_vF(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B2(int(resolution), noise_ratio)
    vF_a = np.round(vF_a, decimals = 4); vF_b = np.round(vF_b, decimals = 4); vF_c = np.round(vF_c, decimals = 4); vF_d = np.round(vF_d, decimals = 4)
    num = 0

    prompt = 'Four datasets containing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Fermi velocity is the gradient of a dispersion at the Fermi energy. Here are three examples with quadratic dispersions. '\
        f'Read "Dataset A". The Fermi velocity of "Dataset A" is {vF_a}. '\
        f'Read "Dataset B". The Fermi velocity of "Dataset B" is {vF_b}. '\
        f'Read "Dataset C". The Fermi velocity of "Dataset C" is {vF_c}. '\
        'Now read "Dataset D", which has a quadratic dispersion. State the Fermi velocity of "Dataset D" as an single number. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B2/B2_vF"
    elif size == 2:
        question_name = "Prompts_med/B2/B2_vF"
    elif size == 3:
        question_name = "Prompts_large/B2/B2_vF"
    else:
        question_name = "Prompts_single/B2/B2_vF"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(vF_d), full_name, "_S")
    else:
        write_to_text(str(vF_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------- B3 ----- Shadow bands (superstructure) --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B3(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 5
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    superstructure_number_a = 5; k_spacing_a = 0.1; attenuation_a = 0.5
    superstructure_params_a = np.array([superstructure_number_a, k_spacing_a, attenuation_a])
    vF_a = linear_band_params_a[0]
    plot_name_a = "Plots/B3/B3a"

    T_b = 120
    gamma_intensity_b = 4
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    superstructure_number_b = 5; k_spacing_b = 0.02; attenuation_b = 0.4
    superstructure_params_b = np.array([superstructure_number_b, k_spacing_b, attenuation_b])
    vF_b = linear_band_params_b[0]
    plot_name_b = "Plots/B3/B3b"

    T_c = 34
    gamma_intensity_c = 8
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    superstructure_number_c = 6; k_spacing_c = 0.04; attenuation_c = 0.45
    superstructure_params_c = np.array([superstructure_number_c, k_spacing_c, attenuation_c])
    vF_c = linear_band_params_c[0]
    plot_name_c = "Plots/B3/B3c"

    T_d = 87
    gamma_intensity_d = 10
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    superstructure_number_d = 7; k_spacing_d = 0.15; attenuation_d = 0.7
    superstructure_params_d = np.array([superstructure_number_d, k_spacing_d, attenuation_d])
    vF_d = linear_band_params_d[0]
    plot_name_d = "Plots/B3/B3d"

    E_array_cut_a, disp_array_cut_a, txt_a = spectrum_FL_superstructure_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, superstructure_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, txt_b = spectrum_FL_superstructure_linear(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, superstructure_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, txt_c = spectrum_FL_superstructure_linear(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, superstructure_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, txt_d = spectrum_FL_superstructure_linear(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, superstructure_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d



def ask_B3_dispersion(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B3(int(resolution), noise_ratio)
    disp_array_cut_a = np.round(disp_array_cut_a, decimals = 4); disp_array_cut_b = np.round(disp_array_cut_b, decimals = 4); disp_array_cut_c = np.round(disp_array_cut_c, decimals = 4); disp_array_cut_d = np.round(disp_array_cut_d, decimals = 4)
    num = len(disp_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. Here are three examples with linear dispersions, overlaid with shadow bands. '\
        'Read "Dataset A". The linear dispersion of "Dataset A" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_a) + ']. '\
        'Read "Dataset B". The linear dispersion of "Dataset B" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_b) + ']. '\
        'Read "Dataset C". The linear dispersion of "Dataset C" is given by the array: [' + ','.join(str(x) for x in disp_array_cut_c) + ']. '\
        f'Now read "Dataset D". Ignoring the shadow bands, state the linear dispersion of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B3/B3"
    elif size == 2:
        question_name = "Prompts_med/B3/B3"
    elif size == 3:
        question_name = "Prompts_large/B3/B3"
    else:
        question_name = "Prompts_single/B3/B3"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(disp_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(disp_array_cut_d), question_name, "_S")

    return question, num



def ask_B3_vF(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, disp_array_cut_a, disp_array_cut_b, disp_array_cut_c, disp_array_cut_d, vF_a, vF_b, vF_c, vF_d, txt_a, txt_b, txt_c, txt_d = get_B3(int(resolution), noise_ratio)
    vF_a = np.round(vF_a, decimals = 4); vF_b = np.round(vF_b, decimals = 4); vF_c = np.round(vF_c, decimals = 4); vF_d = np.round(vF_d, decimals = 4)
    num = 0

    prompt = 'Four datasets containing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Fermi velocity is the gradient of a dispersion at the Fermi energy. Here are three examples with linear dispersions, overlaid with shadow bands. '\
        f'Read "Dataset A". The Fermi velocity of "Dataset A" is {vF_a}. '\
        f'Read "Dataset B". The Fermi velocity of "Dataset B" is {vF_b}. '\
        f'Read "Dataset C". The Fermi velocity of "Dataset C" is {vF_c}. '\
        'Now read "Dataset D", which has a linear dispersion and shadow bands. State the Fermi velocity of "Dataset D" as an single number. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B3/B3_vF"
    elif size == 2:
        question_name = "Prompts_med/B3/B3_vF"
    elif size == 3:
        question_name = "Prompts_large/B3/B3_vF"
    else:
        question_name = "Prompts_single/B3/B3_vF"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(disp_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(disp_array_cut_d), question_name, "_S")

    return question, num





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- B4 ----- Band bottom ----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B4(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params_a = get_band_bottom_params_1_a()
    E0_a = band_bottom_params_a[0]
    plot_name_a = "Plots/B4/B4a"

    T_b = 30
    gamma_intensity_b = 0.6
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params_b = get_band_bottom_params_1_b()
    E0_b = band_bottom_params_b[0]
    plot_name_b = "Plots/B4/B4b"

    T_c = 198
    gamma_intensity_c = 0.5
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params_c = get_band_bottom_params_1_c()
    E0_c = band_bottom_params_c[0]
    plot_name_c = "Plots/B4/B4c"

    T_d = 45
    gamma_intensity_d = 2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params_d = get_band_bottom_params_1_d()
    E0_d = band_bottom_params_d[0]
    plot_name_d = "Plots/B4/B4d"

    k_array_a, Ek_array_cut_a, txt_a = spectrum_FL_band_bottom(spectrum_params_a, smooth_params_a, noise_params_a, band_bottom_params_a, gamma_intensity_a, dynamic_range, plot_name_a)
    k_array_b, Ek_array_cut_b, txt_b = spectrum_FL_band_bottom(spectrum_params_b, smooth_params_b, noise_params_b, band_bottom_params_b, gamma_intensity_b, dynamic_range, plot_name_b)
    k_array_c, Ek_array_cut_c, txt_c = spectrum_FL_band_bottom(spectrum_params_c, smooth_params_c, noise_params_c, band_bottom_params_c, gamma_intensity_c, dynamic_range, plot_name_c)
    k_array_d, Ek_array_cut_d, txt_d = spectrum_FL_band_bottom(spectrum_params_d, smooth_params_d, noise_params_d, band_bottom_params_d, gamma_intensity_d, dynamic_range, plot_name_d)

    return k_array_a, k_array_b, k_array_c, k_array_d, Ek_array_cut_a, Ek_array_cut_b, Ek_array_cut_c, Ek_array_cut_d, E0_a, E0_b, E0_c, E0_d, txt_a, txt_b, txt_c, txt_d



def ask_B4_dispersion(resolution, size, noise_ratio):
    
    k_array_a, k_array_b, k_array_c, k_array_d, Ek_array_cut_a, Ek_array_cut_b, Ek_array_cut_c, Ek_array_cut_d, E0_a, E0_b, E0_c, E0_d, txt_a, txt_b, txt_c, txt_d = get_B4(int(resolution), noise_ratio)
    Ek_array_cut_a = np.round(Ek_array_cut_a, decimals = 4); Ek_array_cut_b = np.round(Ek_array_cut_b, decimals = 4); Ek_array_cut_c = np.round(Ek_array_cut_c, decimals = 4); Ek_array_cut_d = np.round(Ek_array_cut_d, decimals = 4)
    num = len(Ek_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of energies corresponding to the maximum spectral intensity at each momentum. Here are three examples. '\
        'Note that a value of nan corresponds to no dispersion present within the energy range of the spectrum at that momentum. '\
        'Read "Dataset A". The dispersion of "Dataset A" is given by the array: [' + ','.join(str(x) for x in Ek_array_cut_a) + ']. '\
        'Read "Dataset B". The dispersion of "Dataset B" is given by the array: [' + ','.join(str(x) for x in Ek_array_cut_b) + ']. '\
        'Read "Dataset C". The dispersion of "Dataset C" is given by the array: [' + ','.join(str(x) for x in Ek_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the dispersion of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B4/B4"
    elif size == 2:
        question_name = "Prompts_med/B4/B4"
    elif size == 3:
        question_name = "Prompts_large/B4/B4"
    else:
        question_name = "Prompts_single/B4/B4"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(Ek_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(Ek_array_cut_d), question_name, "_S")

    return question, num



def ask_B4_bbE(resolution, size, noise_ratio):
    
    k_array_a, k_array_b, k_array_c, k_array_d, Ek_array_cut_a, Ek_array_cut_b, Ek_array_cut_c, Ek_array_cut_d, E0_a, E0_b, E0_c, E0_d, txt_a, txt_b, txt_c, txt_d = get_B4(int(resolution), noise_ratio)
    E0_a = np.round(E0_a, decimals = 4); E0_b = np.round(E0_b, decimals = 4); E0_c = np.round(E0_c, decimals = 4); E0_d = np.round(E0_d, decimals = 4)
    num = 0

    prompt = 'Four datasets containing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Here are three examples and their corresponding band bottom energies. '\
        f'Read "Dataset A". The band bottom energy of "Dataset A" is {E0_a} eV. '\
        f'Read "Dataset B". The band bottom energy of "Dataset B" is {E0_b} eV. '\
        f'Read "Dataset C". The band bottom energy of "Dataset C" is {E0_c} eV. '\
        'Now read "Dataset D". State the band bottom energy of "Dataset D" in units of electron-Volts. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B4/B4_bbE"
    elif size == 2:
        question_name = "Prompts_med/B4/B4_bbE"
    elif size == 3:
        question_name = "Prompts_large/B4/B4_bbE"
    else:
        question_name = "Prompts_single/B4/B4_bbE"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(E0_d), full_name, "_S")
    else:
        write_to_text(str(E0_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------- B5 ----- Dirac cone -----------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B5(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 2
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    Dirac_cone_params_a = get_Dirac_cone_params_a()
    plot_name_a = "Plots/B5/B5a"

    T_b = 50
    gamma_intensity_b = 1
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    Dirac_cone_params_b = get_Dirac_cone_params_b()
    plot_name_b = "Plots/B5/B5b"

    T_c = 78
    gamma_intensity_c = 3
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    Dirac_cone_params_c = get_Dirac_cone_params_c()
    plot_name_c = "Plots/B5/B5c"

    T_d = 140
    gamma_intensity_d = 2.5
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_a, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    Dirac_cone_params_d = get_Dirac_cone_params_d()
    plot_name_d = "Plots/B5/B5d"

    E0_a, txt_a = spectrum_FL_Dirac(spectrum_params_a, smooth_params_a, noise_params_a, Dirac_cone_params_a, gamma_intensity_a, dynamic_range, plot_name_a)
    E0_b, txt_b = spectrum_FL_Dirac(spectrum_params_b, smooth_params_b, noise_params_b, Dirac_cone_params_b, gamma_intensity_b, dynamic_range, plot_name_b)
    E0_c, txt_c = spectrum_FL_Dirac(spectrum_params_c, smooth_params_c, noise_params_c, Dirac_cone_params_c, gamma_intensity_c, dynamic_range, plot_name_c)
    E0_d, txt_d = spectrum_FL_Dirac(spectrum_params_d, smooth_params_d, noise_params_d, Dirac_cone_params_d, gamma_intensity_d, dynamic_range, plot_name_d)

    return E0_a, E0_b, E0_c, E0_d, txt_a, txt_b, txt_c, txt_d



def ask_B5(resolution, size, noise_ratio):
    
    E0_a, E0_b, E0_c, E0_d, txt_a, txt_b, txt_c, txt_d = get_B5(int(resolution), noise_ratio)
    E0_a = np.round(E0_a, decimals = 4); E0_b = np.round(E0_b, decimals = 4); E0_c = np.round(E0_c, decimals = 4); E0_d = np.round(E0_d, decimals = 4)
    num = 0

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Here are three examples and their corresponding Dirac cone energies. '\
        f'Read "Dataset A". The Dirac cone energy of "Dataset A" is {E0_a} eV. '\
        f'Read "Dataset B". The Dirac cone energy of "Dataset B" is {E0_b} eV. '\
        f'Read "Dataset C". The Dirac cone energy of "Dataset C" is {E0_c} eV. '\
        'Now read "Dataset D". State the Dirac cone energy of "Dataset D" in units of electron-Volts. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B5/B5"
    elif size == 2:
        question_name = "Prompts_med/B5/B5"
    elif size == 3:
        question_name = "Prompts_large/B5/B5"
    else:
        question_name = "Prompts_single/B5/B5"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(E0_d), full_name, "_S")
    else:
        write_to_text(str(E0_d), question_name, "_S")

    return question, num





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------ B6 ----- Superconducting gap ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_B6(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params, k_int_range_a = get_band_linear_params_a()
    BCS_params_a = get_BCS_params_a()
    plot_name_a = "Plots/B6/B6a"

    T_b = 30
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    quadratic_band_params, k_int_range_b = get_band_quadratic_params_a()
    BCS_params_b = get_BCS_params_b()
    plot_name_b = "Plots/B6/B6b"

    T_c = 30
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params = get_band_bottom_params_1_a()
    BCS_params_c = get_BCS_params_c()
    plot_name_c = "Plots/B6/B6c"

    T_d = 30
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    band_bottom_params_2 = get_band_bottom_params_2_a()
    BCS_params_d = get_BCS_params_d()
    plot_name_d = "Plots/B6/B6d"

    Delta_a, txt_a = spectrum_SC_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params, BCS_params_a, dynamic_range, plot_name_a)
    Delta_b, txt_b = spectrum_SC_quadratic(spectrum_params_b, smooth_params_b, noise_params_b, quadratic_band_params, BCS_params_b, dynamic_range, plot_name_b)
    Delta_c, txt_c = spectrum_SC_band_bottom(spectrum_params_c, smooth_params_c, noise_params_c, band_bottom_params, BCS_params_c, dynamic_range, plot_name_c)
    Delta_d, txt_d = spectrum_SC_2_band_bottoms(spectrum_params_d, smooth_params_d, noise_params_d, band_bottom_params, band_bottom_params_2, BCS_params_d, BCS_params_d, dynamic_range, plot_name_d)

    return Delta_a, Delta_b, Delta_c, Delta_d, txt_a, txt_b, txt_c, txt_d



def ask_B6(resolution, size, noise_ratio):
    
    Delta_a, Delta_b, Delta_c, Delta_d, txt_a, txt_b, txt_c, txt_d = get_B6(int(resolution), noise_ratio)
    Delta_a = np.round(Delta_a, decimals = 4); Delta_b = np.round(Delta_b, decimals = 4); Delta_c = np.round(Delta_c, decimals = 4); Delta_d = np.round(Delta_d, decimals = 4)
    num = 0

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Here are three examples and their corresponding superconducting gap energies. '\
        f'Read "Dataset A". The superconducting gap energy of "Dataset A" is {Delta_a} eV. '\
        f'Read "Dataset B". The superconducting gap energy of "Dataset B" is {Delta_b} eV. '\
        f'Read "Dataset C". The superconducting gap energy of "Dataset C" is {Delta_c} eV. '\
        'Now read "Dataset D". State the superconducting gap energy of "Dataset D" in units of electron-Volts. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/B6/B6"
    elif size == 2:
        question_name = "Prompts_med/B6/B6"
    elif size == 3:
        question_name = "Prompts_large/B6/B6"
    else:
        question_name = "Prompts_single/B6/B6"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(Delta_d), full_name, "_S")
    else:
        write_to_text(str(Delta_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- C1 ----- Constant linewidth (impurity scattering) --------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_C1(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    fwhm_a = 0.05
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    plot_name_a = "Plots/C1/C1a"

    T_b = 30
    fwhm_b = 0.025
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    plot_name_b = "Plots/C1/C1b"

    T_c = 200
    fwhm_c = 0.07
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    plot_name_c = "Plots/C1/C1c"

    T_d = 200
    fwhm_d = 0.15
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    plot_name_d = "Plots/C1/C1d"

    E_array_cut_a, disp_array_cut_a, gamma_array_cut_a, txt_a = spectrum_imp_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, fwhm_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, gamma_array_cut_b, txt_b = spectrum_imp_linear(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, fwhm_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, gamma_array_cut_c, txt_c = spectrum_imp_linear(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, fwhm_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, gamma_array_cut_d, txt_d = spectrum_imp_linear(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, fwhm_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d



def ask_C1(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d = get_C1(int(resolution), noise_ratio)
    gamma_array_cut_a = np.round(gamma_array_cut_a, decimals = 4); gamma_array_cut_b = np.round(gamma_array_cut_b, decimals = 4); gamma_array_cut_c = np.round(gamma_array_cut_c, decimals = 4); gamma_array_cut_d = np.round(gamma_array_cut_d, decimals = 4)
    num = len(gamma_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'Here are three examples of widths corresponding to different dispersions. '\
        'Read "Dataset A". The width of "Dataset A" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_a) + ']. '\
        'Read "Dataset B". The width of "Dataset B" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_b) + ']. '\
        'Read "Dataset C". The width of "Dataset C" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the width of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/C1/C1"
    elif size == 2:
        question_name = "Prompts_med/C1/C1"
    elif size == 3:
        question_name = "Prompts_large/C1/C1"
    else:
        question_name = "Prompts_single/C1/C1"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(gamma_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(gamma_array_cut_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- C2 ----- Linear linewidth (Marginal Fermi liquid) --------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_C2(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 0.3
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    plot_name_a = "Plots/C2/C2a"

    T_b = 100
    gamma_intensity_b = 0.2
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    plot_name_b = "Plots/C2/C2b"

    T_c = 57
    gamma_intensity_c = 0.2
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    plot_name_c = "Plots/C2/C2c"

    T_d = 25
    gamma_intensity_d = 0.2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    plot_name_d = "Plots/C2/C2d"

    E_array_cut_a, disp_array_cut_a, gamma_array_cut_a, txt_a = spectrum_MFL_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, gamma_array_cut_b, txt_b = spectrum_MFL_linear(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, gamma_array_cut_c, txt_c = spectrum_MFL_linear(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, gamma_array_cut_d, txt_d = spectrum_MFL_linear(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d



def ask_C2(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d = get_C2(int(resolution), noise_ratio)
    gamma_array_cut_a = np.round(gamma_array_cut_a, decimals = 4); gamma_array_cut_b = np.round(gamma_array_cut_b, decimals = 4); gamma_array_cut_c = np.round(gamma_array_cut_c, decimals = 4); gamma_array_cut_d = np.round(gamma_array_cut_d, decimals = 4)
    num = len(gamma_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'Here are three examples of widths corresponding to different dispersions. '\
        'Read "Dataset A". The width of "Dataset A" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_a) + ']. '\
        'Read "Dataset B". The width of "Dataset B" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_b) + ']. '\
        'Read "Dataset C". The width of "Dataset C" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the width of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/C2/C2"
    elif size == 2:
        question_name = "Prompts_med/C2/C2"
    elif size == 3:
        question_name = "Prompts_large/C2/C2"
    else:
        question_name = "Prompts_single/C2/C2"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(gamma_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(gamma_array_cut_d), question_name, "_S")

    return question, num




# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------- C3 ----- Quadratic linewidth (Fermi liquid) -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_C3(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    plot_name_a = "Plots/C3/C3a"

    T_b = 130
    gamma_intensity_b = 1.5
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    plot_name_b = "Plots/C3/C3b"

    T_c = 167
    gamma_intensity_c = 1.7
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    plot_name_c = "Plots/C3/C3c"

    T_d = 183
    gamma_intensity_d = 2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    plot_name_d = "Plots/C3/C3d"

    E_array_cut_a, disp_array_cut_a, gamma_array_cut_a, txt_a = spectrum_FL_linear(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, disp_array_cut_b, gamma_array_cut_b, txt_b = spectrum_FL_linear(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, disp_array_cut_c, gamma_array_cut_c, txt_c = spectrum_FL_linear(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, disp_array_cut_d, gamma_array_cut_d, txt_d = spectrum_FL_linear(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d



def ask_C3(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d = get_C3(int(resolution), noise_ratio)
    gamma_array_cut_a = np.round(gamma_array_cut_a, decimals = 4); gamma_array_cut_b = np.round(gamma_array_cut_b, decimals = 4); gamma_array_cut_c = np.round(gamma_array_cut_c, decimals = 4); gamma_array_cut_d = np.round(gamma_array_cut_d, decimals = 4)
    num = len(gamma_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'Here are three examples of widths corresponding to different dispersions. '\
        'Read "Dataset A". The width of "Dataset A" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_a) + ']. '\
        'Read "Dataset B". The width of "Dataset B" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_b) + ']. '\
        'Read "Dataset C". The width of "Dataset C" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the width of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/C3/C3"
    elif size == 2:
        question_name = "Prompts_med/C3/C3"
    elif size == 3:
        question_name = "Prompts_large/C3/C3"
    else:
        question_name = "Prompts_single/C3/C3"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(gamma_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(gamma_array_cut_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- C4 ----- Mixed linewidths: 1 phonon + Fermi liquid --------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_C4(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000
    coupling_lambda = 1

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    phonon_params_a = np.concatenate((coupling_lambda, get_1_phonon_a()), axis = None)
    plot_name_a = "Plots/C4/C4a"

    T_b = 132
    gamma_intensity_b = 0.5
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    phonon_params_b = np.concatenate((coupling_lambda, get_1_phonon_b()), axis = None)
    plot_name_b = "Plots/C4/C4b"

    T_c = 72
    gamma_intensity_c = 0.3
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    phonon_params_c = np.concatenate((coupling_lambda, get_1_phonon_c()), axis = None)
    plot_name_c = "Plots/C4/C4c"

    T_d = 96
    gamma_intensity_d = 0.2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    phonon_params_d = np.concatenate((coupling_lambda, get_1_phonon_d()), axis = None)
    plot_name_d = "Plots/C4/C4d"

    E_array_cut_a, hn_a, disp_array_cut_a, k_phonon_cut_a, ReS_array_cut_a, ImS_array_cut_a, txt_a = spectrum_phonon_1_FL(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, phonon_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, hn_b, disp_array_cut_b, k_phonon_cut_b, ReS_array_cut_b, ImS_array_cut_b, txt_b = spectrum_phonon_1_FL(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, phonon_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, hn_c, disp_array_cut_c, k_phonon_cut_c, ReS_array_cut_c, ImS_array_cut_c, txt_c = spectrum_phonon_1_FL(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, phonon_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, hn_d, disp_array_cut_d, k_phonon_cut_d, ReS_array_cut_d, ImS_array_cut_d, txt_d = spectrum_phonon_1_FL(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, phonon_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, ImS_array_cut_a, ImS_array_cut_b, ImS_array_cut_c, ImS_array_cut_d, txt_a, txt_b, txt_c, txt_d



def ask_C4(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d = get_C4(int(resolution), noise_ratio)
    gamma_array_cut_a = np.round(gamma_array_cut_a, decimals = 4); gamma_array_cut_b = np.round(gamma_array_cut_b, decimals = 4); gamma_array_cut_c = np.round(gamma_array_cut_c, decimals = 4); gamma_array_cut_d = np.round(gamma_array_cut_d, decimals = 4)
    num = len(gamma_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'Here are three examples of widths corresponding to different dispersions. '\
        'Read "Dataset A". The width of "Dataset A" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_a) + ']. '\
        'Read "Dataset B". The width of "Dataset B" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_b) + ']. '\
        'Read "Dataset C". The width of "Dataset C" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the width of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/C4/C4"
    elif size == 2:
        question_name = "Prompts_med/C4/C4"
    elif size == 3:
        question_name = "Prompts_large/C4/C4"
    else:
        question_name = "Prompts_single/C4/C4"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(gamma_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(gamma_array_cut_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- C5 ----- Mixed linewidths: 1 phonon + Marginal Fermi liquid ---------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_C5(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000
    coupling_lambda = 0.5

    T_a = 300
    gamma_intensity_a = 0.3
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    phonon_params_a = np.concatenate((coupling_lambda, get_1_phonon_a()), axis = None)
    plot_name_a = "Plots/C5/C5a"

    T_b = 42
    gamma_intensity_b = 0.2
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    phonon_params_b = np.concatenate((coupling_lambda, get_1_phonon_b()), axis = None)
    plot_name_b = "Plots/C5/C5b"

    T_c = 175
    gamma_intensity_c = 0.15
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    phonon_params_c = np.concatenate((coupling_lambda, get_1_phonon_c()), axis = None)
    plot_name_c = "Plots/C5/C5c"

    T_d = 89
    gamma_intensity_d = 0.1
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    phonon_params_d = np.concatenate((coupling_lambda, get_1_phonon_d()), axis = None)
    plot_name_d = "Plots/C5/C5d"

    E_array_cut_a, hn_a, disp_array_cut_a, k_phonon_cut_a, ReS_array_cut_a, ImS_array_cut_a, txt_a = spectrum_phonon_1_MFL(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, phonon_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, hn_b, disp_array_cut_b, k_phonon_cut_b, ReS_array_cut_b, ImS_array_cut_b, txt_b = spectrum_phonon_1_MFL(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, phonon_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, hn_c, disp_array_cut_c, k_phonon_cut_c, ReS_array_cut_c, ImS_array_cut_c, txt_c = spectrum_phonon_1_MFL(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, phonon_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, hn_d, disp_array_cut_d, k_phonon_cut_d, ReS_array_cut_d, ImS_array_cut_d, txt_d = spectrum_phonon_1_MFL(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, phonon_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, ImS_array_cut_a, ImS_array_cut_b, ImS_array_cut_c, ImS_array_cut_d, txt_a, txt_b, txt_c, txt_d



def ask_C5(resolution, size, noise_ratio):
    
    E_array_cut_a, E_array_cut_b, E_array_cut_c, E_array_cut_d, gamma_array_cut_a, gamma_array_cut_b, gamma_array_cut_c, gamma_array_cut_d, txt_a, txt_b, txt_c, txt_d = get_C5(int(resolution), noise_ratio)
    gamma_array_cut_a = np.round(gamma_array_cut_a, decimals = 4); gamma_array_cut_b = np.round(gamma_array_cut_b, decimals = 4); gamma_array_cut_c = np.round(gamma_array_cut_c, decimals = 4); gamma_array_cut_d = np.round(gamma_array_cut_d, decimals = 4)
    num = len(gamma_array_cut_d)

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'Here are three examples of widths corresponding to different dispersions. '\
        'Read "Dataset A". The width of "Dataset A" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_a) + ']. '\
        'Read "Dataset B". The width of "Dataset B" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_b) + ']. '\
        'Read "Dataset C". The width of "Dataset C" is given by the array: [' + ','.join(str(x) for x in gamma_array_cut_c) + ']. '\
        f'Now read "Dataset D". State the width of "Dataset D" as an array of {num} numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/C5/C5"
    elif size == 2:
        question_name = "Prompts_med/C5/C5"
    elif size == 3:
        question_name = "Prompts_large/C5/C5"
    else:
        question_name = "Prompts_single/C5/C5"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(gamma_array_cut_d), full_name, "_S")
    else:
        write_to_text(str(gamma_array_cut_d), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- D1 ----- Renormalization: 1 phonon + Fermi liquid --------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_D1(resolution, noise_ratio, coupling_lambda):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_extended_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    phonon_params_a = np.concatenate((coupling_lambda, get_1_phonon_a()), axis = None)
    plot_name_a = "Plots/D1/D1a"

    T_b = 10
    gamma_intensity_b = 0.5
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_extended_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    phonon_params_b = np.concatenate((coupling_lambda, get_1_phonon_b()), axis = None)
    plot_name_b = "Plots/D1/D1b"

    T_c = 25
    gamma_intensity_c = 0.5
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_extended_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    phonon_params_c = np.concatenate((coupling_lambda, get_1_phonon_c()), axis = None)
    plot_name_c = "Plots/D1/D1c"

    T_d = 125
    gamma_intensity_d = 2
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_extended_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    phonon_params_d = np.concatenate((coupling_lambda, get_1_phonon_d()), axis = None)
    plot_name_d = "Plots/D1/D1d"

    E_array_cut_a, hn_a, disp_array_cut_a, k_phonon_cut_a, ReS_array_cut_a, ImS_array_cut_a, txt_a = spectrum_phonon_1_FL(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, phonon_params_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, hn_b, disp_array_cut_b, k_phonon_cut_b, ReS_array_cut_b, ImS_array_cut_b, txt_b = spectrum_phonon_1_FL(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, phonon_params_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, hn_c, disp_array_cut_c, k_phonon_cut_c, ReS_array_cut_c, ImS_array_cut_c, txt_c = spectrum_phonon_1_FL(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, phonon_params_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, hn_d, disp_array_cut_d, k_phonon_cut_d, ReS_array_cut_d, ImS_array_cut_d, txt_d = spectrum_phonon_1_FL(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, phonon_params_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return hn_a, hn_b, hn_c, hn_d, txt_a, txt_b, txt_c, txt_d



def ask_D1(resolution, size, noise_ratio, coupling_lambda):
    
    hn_a, hn_b, hn_c, hn_d, txt_a, txt_b, txt_c, txt_d = get_D1(int(resolution), noise_ratio, coupling_lambda)
    hn_a = np.round(hn_a, decimals = 4); hn_b = np.round(hn_b, decimals = 4); hn_c = np.round(hn_c, decimals = 4); hn_d = np.round(hn_d, decimals = 4)
    num = 0

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'The presence of a phonon may be deduced from a kink in the dispersion at some energy, and a corresponding increase in width below that energy. '\
        'That energy is therefore taken to be the phonon energy. Here are three examples. '\
        f'Read "Dataset A". The phonon energy of "Dataset A" is {hn_a} eV. '\
        f'Read "Dataset B". The phonon energy of "Dataset B" is {hn_b} eV. '\
        f'Read "Dataset C". The phonon energy of "Dataset C" is {hn_c} eV. '\
        'Now read "Dataset D". State the phonon energy of "Dataset D" in units of electron-Volts. Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/D1/D1"
    elif size == 2:
        question_name = "Prompts_med/D1/D1"
    elif size == 3:
        question_name = "Prompts_large/D1/D1"
    else:
        question_name = "Prompts_single/D1/D1"

    noise_int = round(100*noise_ratio); lambda_val = str(coupling_lambda)
    lambda_val = lambda_val.replace(".", "")
    append_parameters = "_L" + lambda_val + f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(hn_d), full_name, "_S")
    else:
        write_to_text(str(hn_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- D2 ----- Renormalization: 2 phonons + Fermi liquid --------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_D2(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_extended_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    coupling_lambda_1_a = 0.8; coupling_lambda_2_a = 1.2
    phonon_params_1_a_short, phonon_params_2_a_short = get_2_phonons_a()
    phonon_params_1_a = np.concatenate((coupling_lambda_1_a, phonon_params_1_a_short), axis = None)
    phonon_params_2_a = np.concatenate((coupling_lambda_2_a, phonon_params_2_a_short), axis = None)
    plot_name_a = "Plots/D2/D2a"

    T_b = 10
    gamma_intensity_b = 1
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_extended_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    coupling_lambda_1_b = 1; coupling_lambda_2_b = 1
    phonon_params_1_b_short, phonon_params_2_b_short = get_2_phonons_b()
    phonon_params_1_b = np.concatenate((coupling_lambda_1_b, phonon_params_1_b_short), axis = None)
    phonon_params_2_b = np.concatenate((coupling_lambda_2_b, phonon_params_2_b_short), axis = None)
    plot_name_b = "Plots/D2/D2b"

    T_c = 70
    gamma_intensity_c = 1
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_extended_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    coupling_lambda_1_c = 1.3; coupling_lambda_2_c = 0.8
    phonon_params_1_c_short, phonon_params_2_c_short = get_2_phonons_c()
    phonon_params_1_c = np.concatenate((coupling_lambda_1_c, phonon_params_1_c_short), axis = None)
    phonon_params_2_c = np.concatenate((coupling_lambda_2_c, phonon_params_2_c_short), axis = None)
    plot_name_c = "Plots/D2/D2c"

    T_d = 35
    gamma_intensity_d = 1
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_extended_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    coupling_lambda_1_d = 0.6; coupling_lambda_2_d = 1.5
    phonon_params_1_d_short, phonon_params_2_d_short = get_2_phonons_d()
    phonon_params_1_d = np.concatenate((coupling_lambda_1_d, phonon_params_1_d_short), axis = None)
    phonon_params_2_d = np.concatenate((coupling_lambda_2_d, phonon_params_2_d_short), axis = None)
    plot_name_d = "Plots/D2/D2d"

    E_array_cut_a, hn_1_a, hn_2_a, disp_array_cut_a, k_phonon_cut_a, ReS_array_cut_a, ImS_array_cut_a, txt_a = spectrum_phonon_2_FL(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, phonon_params_1_a, phonon_params_2_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, hn_1_b, hn_2_b, disp_array_cut_b, k_phonon_cut_b, ReS_array_cut_b, ImS_array_cut_b, txt_b = spectrum_phonon_2_FL(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, phonon_params_1_b, phonon_params_2_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, hn_1_c, hn_2_c, disp_array_cut_c, k_phonon_cut_c, ReS_array_cut_c, ImS_array_cut_c, txt_c = spectrum_phonon_2_FL(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, phonon_params_1_c, phonon_params_2_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, hn_1_d, hn_2_d, disp_array_cut_d, k_phonon_cut_d, ReS_array_cut_d, ImS_array_cut_d, txt_d = spectrum_phonon_2_FL(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, phonon_params_1_d, phonon_params_2_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return hn_1_a, hn_2_a, hn_1_b, hn_2_b, hn_1_c, hn_2_c, hn_1_d, hn_2_d, txt_a, txt_b, txt_c, txt_d



def ask_D2(resolution, size, noise_ratio):
    
    hn_1_a, hn_2_a, hn_1_b, hn_2_b, hn_1_c, hn_2_c, hn_1_d, hn_2_d, txt_a, txt_b, txt_c, txt_d = get_D2(int(resolution), noise_ratio)
    hn_a = np.array([hn_1_a, hn_2_a]); hn_b = np.array([hn_1_b, hn_2_b]); hn_c = np.array([hn_1_c, hn_2_c]); hn_d = np.array([hn_1_d, hn_2_d])
    hn_a = np.round(hn_a, decimals = 4); hn_b = np.round(hn_b, decimals = 4); hn_c = np.round(hn_c, decimals = 4); hn_d = np.round(hn_d, decimals = 4)
    num = 2

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'The presence of a phonon may be deduced from a kink in the dispersion at some energy, and a corresponding increase in width below that energy. '\
        'That energy is therefore taken to be the phonon energy. Here are three examples of spectra with two phonons each. Their energies are stated in decreasing value, and in the form of arrays. '\
        'Read "Dataset A". The phonon energies of "Dataset A" are [' + ','.join(str(x) for x in hn_a) + '] eV. '\
        'Read "Dataset B". The phonon energies of "Dataset B" are [' + ','.join(str(x) for x in hn_b) + '] eV. '\
        'Read "Dataset C". The phonon energies of "Dataset C" are [' + ','.join(str(x) for x in hn_c) + '] eV. '\
        'Now read "Dataset D". State the two phonon energies of "Dataset D" as an array of two numbers in decreasing value. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/D2/D2"
    elif size == 2:
        question_name = "Prompts_med/D2/D2"
    elif size == 3:
        question_name = "Prompts_large/D2/D2"
    else:
        question_name = "Prompts_single/D2/D2"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(hn_d), full_name, "_S")
    else:
        write_to_text(str(hn_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- D3 ----- Renormalization: 3 phonons + Fermi liquid --------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_D3(resolution, noise_ratio):

    k_conv = 0.005; E_conv = 0.003
    dynamic_range = 1000

    T_a = 300
    gamma_intensity_a = 1
    spectrum_params_a = np.concatenate((resolution, get_spectrum_params_extended_a()), axis = None)
    smooth_params_a = np.array([T_a, k_conv, E_conv])
    noise_params_a = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_a, k_int_range_a = get_band_linear_params_a()
    coupling_lambda_1_a = 0.7; coupling_lambda_2_a = 1; coupling_lambda_3_a = 1.2
    phonon_params_1_a_short, phonon_params_2_a_short, phonon_params_3_a_short = get_3_phonons_a()
    phonon_params_1_a = np.concatenate((coupling_lambda_1_a, phonon_params_1_a_short), axis = None)
    phonon_params_2_a = np.concatenate((coupling_lambda_2_a, phonon_params_2_a_short), axis = None)
    phonon_params_3_a = np.concatenate((coupling_lambda_3_a, phonon_params_3_a_short), axis = None)
    plot_name_a = "Plots/D3/D3a"

    T_b = 127
    gamma_intensity_b = 0.8
    spectrum_params_b = np.concatenate((resolution, get_spectrum_params_extended_b()), axis = None)
    smooth_params_b = np.array([T_b, k_conv, E_conv])
    noise_params_b = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_b, k_int_range_b = get_band_linear_params_b()
    coupling_lambda_1_b = 0.8; coupling_lambda_2_b = 1.2; coupling_lambda_3_b = 0.9
    phonon_params_1_b_short, phonon_params_2_b_short, phonon_params_3_b_short = get_3_phonons_b()
    phonon_params_1_b = np.concatenate((coupling_lambda_1_b, phonon_params_1_b_short), axis = None)
    phonon_params_2_b = np.concatenate((coupling_lambda_2_b, phonon_params_2_b_short), axis = None)
    phonon_params_3_b = np.concatenate((coupling_lambda_3_b, phonon_params_3_b_short), axis = None)
    plot_name_b = "Plots/D3/D3b"

    T_c = 6
    gamma_intensity_c = 1.2
    spectrum_params_c = np.concatenate((resolution, get_spectrum_params_extended_c()), axis = None)
    smooth_params_c = np.array([T_c, k_conv, E_conv])
    noise_params_c = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_c, k_int_range_c = get_band_linear_params_c()
    coupling_lambda_1_c = 1; coupling_lambda_2_c = 0.8; coupling_lambda_3_c = 1.2
    phonon_params_1_c_short, phonon_params_2_c_short, phonon_params_3_c_short = get_3_phonons_c()
    phonon_params_1_c = np.concatenate((coupling_lambda_1_c, phonon_params_1_c_short), axis = None)
    phonon_params_2_c = np.concatenate((coupling_lambda_2_c, phonon_params_2_c_short), axis = None)
    phonon_params_3_c = np.concatenate((coupling_lambda_3_c, phonon_params_3_c_short), axis = None)
    plot_name_c = "Plots/D3/D3c"

    T_d = 25
    gamma_intensity_d = 0.3
    spectrum_params_d = np.concatenate((resolution, get_spectrum_params_extended_d()), axis = None)
    smooth_params_d = np.array([T_d, k_conv, E_conv])
    noise_params_d = np.concatenate((get_default_noise(), noise_ratio), axis = None)
    linear_band_params_d, k_int_range_d = get_band_linear_params_d()
    coupling_lambda_1_d = 0.7; coupling_lambda_2_d = 0.6; coupling_lambda_3_d = 0.4
    phonon_params_1_d_short, phonon_params_2_d_short, phonon_params_3_d_short = get_3_phonons_d()
    phonon_params_1_d = np.concatenate((coupling_lambda_1_d, phonon_params_1_d_short), axis = None)
    phonon_params_2_d = np.concatenate((coupling_lambda_2_d, phonon_params_2_d_short), axis = None)
    phonon_params_3_d = np.concatenate((coupling_lambda_3_d, phonon_params_3_d_short), axis = None)
    plot_name_d = "Plots/D3/D3d"

    E_array_cut_a, hn_1_a, hn_2_a, hn_3_a, disp_array_cut_a, k_phonon_cut_a, ReS_array_cut_a, ImS_array_cut_a, txt_a = spectrum_phonon_3_FL(spectrum_params_a, smooth_params_a, noise_params_a, linear_band_params_a, phonon_params_1_a, phonon_params_2_a, phonon_params_3_a, gamma_intensity_a, dynamic_range, k_int_range_a, plot_name_a)
    E_array_cut_b, hn_1_b, hn_2_b, hn_3_b, disp_array_cut_b, k_phonon_cut_b, ReS_array_cut_b, ImS_array_cut_b, txt_b = spectrum_phonon_3_FL(spectrum_params_b, smooth_params_b, noise_params_b, linear_band_params_b, phonon_params_1_b, phonon_params_2_b, phonon_params_3_b, gamma_intensity_b, dynamic_range, k_int_range_b, plot_name_b)
    E_array_cut_c, hn_1_c, hn_2_c, hn_3_c, disp_array_cut_c, k_phonon_cut_c, ReS_array_cut_c, ImS_array_cut_c, txt_c = spectrum_phonon_3_FL(spectrum_params_c, smooth_params_c, noise_params_c, linear_band_params_c, phonon_params_1_c, phonon_params_2_c, phonon_params_3_c, gamma_intensity_c, dynamic_range, k_int_range_c, plot_name_c)
    E_array_cut_d, hn_1_d, hn_2_d, hn_3_d, disp_array_cut_d, k_phonon_cut_d, ReS_array_cut_d, ImS_array_cut_d, txt_d = spectrum_phonon_3_FL(spectrum_params_d, smooth_params_d, noise_params_d, linear_band_params_d, phonon_params_1_d, phonon_params_2_d, phonon_params_3_d, gamma_intensity_d, dynamic_range, k_int_range_d, plot_name_d)

    return hn_1_a, hn_2_a, hn_3_a, hn_1_b, hn_2_b, hn_3_b, hn_1_c, hn_2_c, hn_3_c, hn_1_d, hn_2_d, hn_3_d, txt_a, txt_b, txt_c, txt_d



def ask_D3(resolution, size, noise_ratio):
    
    hn_1_a, hn_2_a, hn_3_a, hn_1_b, hn_2_b, hn_3_b, hn_1_c, hn_2_c, hn_3_c, hn_1_d, hn_2_d, hn_3_d, txt_a, txt_b, txt_c, txt_d = get_D3(int(resolution), noise_ratio)
    hn_a = np.array([hn_1_a, hn_2_a, hn_3_a]); hn_b = np.array([hn_1_b, hn_2_b, hn_3_b]); hn_c = np.array([hn_1_c, hn_2_c, hn_3_c]); hn_d = np.array([hn_1_d, hn_2_d, hn_3_d])
    hn_a = np.round(hn_a, decimals = 4); hn_b = np.round(hn_b, decimals = 4); hn_c = np.round(hn_c, decimals = 4); hn_d = np.round(hn_d, decimals = 4)
    num = 3

    prompt = 'Four datasets showing ARPES spectra are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'A dispersion is the set of momenta corresponding to the maximum spectral intensity at each energy. '\
        'The width of a dispersion is defined by how separated (in momentum) the points of half-maximum intensity are, at each energy, after accounting for noise and convolution. '\
        'The presence of a phonon may be deduced from a kink in the dispersion at some energy, and a corresponding increase in width below that energy. '\
        'That energy is therefore taken to be the phonon energy. Here are three examples of spectra with three phonons each. Their energies are stated in decreasing value, and in the form of arrays. '\
        'Read "Dataset A". The phonon energies of "Dataset A" are [' + ','.join(str(x) for x in hn_a) + '] eV. '\
        'Read "Dataset B". The phonon energies of "Dataset B" are [' + ','.join(str(x) for x in hn_b) + '] eV. '\
        'Read "Dataset C". The phonon energies of "Dataset C" are [' + ','.join(str(x) for x in hn_c) + '] eV. '\
        'Now read "Dataset D". State the three phonon energies of "Dataset D" as an array of three numbers in decreasing value. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/D3/D3"
    elif size == 2:
        question_name = "Prompts_med/D3/D3"
    elif size == 3:
        question_name = "Prompts_large/D3/D3"
    else:
        question_name = "Prompts_single/D3/D3"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(hn_d), full_name, "_S")
    else:
        write_to_text(str(hn_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------- E1 ----- Fermi surface: 1-band cuprate --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_E1(resolution, noise_ratio):

    dynamic_range = 1000
    E0 = 0; E_conv = 0.0001; dE = 0.0001
    E_params = np.array([E0, E_conv, dE])
    xy_bound = 1; bkg = 0

    map_params = np.array([resolution, xy_bound, bkg])

    k_conv_a = 0.06
    coefs_a = get_cuprate_monolayer_coefs_a()
    noise_params_a = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_a = "Plots/E1/E1a"

    k_conv_b = 0.055
    coefs_b = get_cuprate_monolayer_coefs_b()
    noise_params_b = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_b = "Plots/E1/E1b"

    k_conv_c = 0.07
    coefs_c = get_cuprate_monolayer_coefs_c()
    noise_params_c = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_c = "Plots/E1/E1c"

    k_conv_d = 0.05
    coefs_d = get_cuprate_monolayer_coefs_d()
    noise_params_d = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_d = "Plots/E1/E1d"

    doping_a, txt_a = map_cuprate_monolayer_band(map_params, E_params, noise_params_a, k_conv_a, coefs_a, dynamic_range, plot_name_a)
    doping_b, txt_b = map_cuprate_monolayer_band(map_params, E_params, noise_params_b, k_conv_b, coefs_b, dynamic_range, plot_name_b)
    doping_c, txt_c = map_cuprate_monolayer_band(map_params, E_params, noise_params_c, k_conv_c, coefs_c, dynamic_range, plot_name_c)
    doping_d, txt_d = map_cuprate_monolayer_band(map_params, E_params, noise_params_d, k_conv_d, coefs_d, dynamic_range, plot_name_d)

    return doping_a, doping_b, doping_c, doping_d, txt_a, txt_b, txt_c, txt_d



def ask_E1(resolution, size, noise_ratio):
    
    doping_a, doping_b, doping_c, doping_d, txt_a, txt_b, txt_c, txt_d = get_E1(int(resolution), noise_ratio)
    doping_a = np.round(doping_a, decimals = 4); doping_b = np.round(doping_b, decimals = 4); doping_c = np.round(doping_c, decimals = 4); doping_d = np.round(doping_d, decimals = 4)
    num = 0

    prompt = 'Four datasets showing ARPES Fermi surfaces in one Brillouin zone are contained. They are labelled "Dataset A", "Dataset B", "Dataset C", and "Dataset D". '\
        'Doping level is linearly related to the area enclosed by a Fermi surface. '\
        'If the Fermi surface completely encloses the Brillouin zone, it has a doping level of +1. '\
        'If the Fermi surface encloses none of the Brillouin zone, it has a doping level of -1. '\
        'Note that because Brillouin zones have periodic boundaries, the inside and outside of a Brillouin zone may be swapped. Here are three examples. '\
        f'Read "Dataset A". The doping level of "Dataset A" is {doping_a}. '\
        f'Read "Dataset B". The doping level of "Dataset B" is {doping_b}. '\
        f'Read "Dataset C". The doping level of "Dataset C" is {doping_c}. '\
        'Now read "Dataset D". State the doping level of "Dataset D". Print only your numerical answer.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b + "\n\n Dataset C\n" + txt_c + "\n\n Dataset D\n" + txt_d

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/E1/E1"
    elif size == 2:
        question_name = "Prompts_med/E1/E1"
    elif size == 3:
        question_name = "Prompts_large/E1/E1"
    else:
        question_name = "Prompts_single/E1/E1"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(doping_d), full_name, "_S")
    else:
        write_to_text(str(doping_d), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------- E2 ----- Fermi surface: 2-band cuprate --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_E2(resolution, noise_ratio):

    dynamic_range = 1000
    E0 = 0; E_conv = 0.0001; dE = 0.0001
    E_params = np.array([E0, E_conv, dE])
    xy_bound = 1; bkg = 0

    map_params = np.array([resolution, xy_bound, bkg])

    k_conv_a = 0.04
    coefs_1_a, coefs_2_a = get_cuprate_bilayer_coefs_a()
    noise_params_a = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_a = "Plots/E2/E2a"

    k_conv_b = 0.035
    coefs_1_b, coefs_2_b = get_cuprate_bilayer_coefs_b()
    noise_params_b = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_b = "Plots/E2/E2b"

    doping_1_a, doping_2_a, txt_a = map_cuprate_bilayer_bands(map_params, E_params, noise_params_a, k_conv_a, coefs_1_a, coefs_2_a, dynamic_range, plot_name_a)
    doping_1_b, doping_2_b, txt_b = map_cuprate_bilayer_bands(map_params, E_params, noise_params_b, k_conv_b, coefs_1_b, coefs_2_b, dynamic_range, plot_name_b)

    return doping_1_a, doping_2_a, doping_1_b, doping_2_b, txt_a, txt_b


def ask_E2(resolution, size, noise_ratio):
    
    doping_1_a, doping_2_a, doping_1_b, doping_2_b, txt_a, txt_b = get_E2(int(resolution), noise_ratio)
    doping_a = np.array([doping_1_a, doping_2_a]); doping_b = np.array([doping_1_b, doping_2_b])
    doping_a = np.round(doping_a, decimals = 4); doping_b = np.round(doping_b, decimals = 4)
    num = 2

    prompt = 'Two datasets showing ARPES Fermi surfaces in one Brillouin zone are contained. They are labelled "Dataset A" and "Dataset B". '\
        'Doping level is linearly related to the area enclosed by a Fermi surface. '\
        'If the Fermi surface completely encloses the Brillouin zone, it has a doping level of +1. '\
        'If the Fermi surface encloses none of the Brillouin zone, it has a doping level of -1. '\
        'Note that because Brillouin zones have periodic boundaries, the inside and outside of a Brillouin zone may be swapped. '\
        'Here is an example with two Fermi surfaces, each with its own doping level. '\
        'Read "Dataset A". The doping levels of the two Fermi surfaces "Dataset A" are listed in order as an array; [' + ','.join(str(x) for x in doping_a) + ']. '\
        'Now read "Dataset B". State the doping levels of "Dataset B", in the same order corresponding to each band, as an array of two numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/E2/E2"
    elif size == 2:
        question_name = "Prompts_med/E2/E2"
    elif size == 3:
        question_name = "Prompts_large/E2/E2"
    else:
        question_name = "Prompts_single/E2/E2"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(doping_b), full_name, "_S")
    else:
        write_to_text(str(doping_b), question_name, "_S")

    return question, num





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------- E3 ----- Fermi surface: strontium ruthenate -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_E3(resolution, noise_ratio):

    dynamic_range = 1000
    E0 = 0; E_conv = 0.0001; dE = 0.0001
    E_params = np.array([E0, E_conv, dE])
    xy_bound = 1; bkg = 0

    map_params = np.array([resolution, xy_bound, bkg])

    k_conv_a = 0.03
    coefs_1_a, coefs_2_a, coefs_3_a = get_SRO_coefs_a()
    noise_params_a = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_a = "Plots/E3/E3a"

    k_conv_b = 0.04
    coefs_1_b, coefs_2_b, coefs_3_b = get_SRO_coefs_b()
    noise_params_b = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_b = "Plots/E3/E3b"

    doping_1_a, doping_2_a, doping_3_a, txt_a = map_SRO_bands(map_params, E_params, noise_params_a, k_conv_a, coefs_1_a, coefs_2_a, coefs_3_a, dynamic_range, plot_name_a)
    doping_1_b, doping_2_b, doping_3_b, txt_b = map_SRO_bands(map_params, E_params, noise_params_b, k_conv_b, coefs_1_b, coefs_2_b, coefs_3_b, dynamic_range, plot_name_b)

    return doping_1_a, doping_2_a, doping_3_a, doping_1_b, doping_2_b, doping_3_b, txt_a, txt_b



def ask_E3(resolution, size, noise_ratio):
    
    doping_1_a, doping_2_a, doping_3_a, doping_1_b, doping_2_b, doping_3_b, txt_a, txt_b = get_E3(int(resolution), noise_ratio)
    doping_a = np.array([doping_1_a, doping_2_a, doping_3_a]); doping_b = np.array([doping_1_b, doping_2_b, doping_3_b])
    doping_a = np.round(doping_a, decimals = 4); doping_b = np.round(doping_b, decimals = 4)
    num = 3

    prompt = 'Two datasets showing ARPES Fermi surfaces in one Brillouin zone are contained. They are labelled "Dataset A" and "Dataset B". '\
        'Doping level is linearly related to the area enclosed by a Fermi surface. '\
        'If the Fermi surface completely encloses the Brillouin zone, it has a doping level of +1. '\
        'If the Fermi surface encloses none of the Brillouin zone, it has a doping level of -1. '\
        'Note that because Brillouin zones have periodic boundaries, the inside and outside of a Brillouin zone may be swapped. '\
        'Here is an example with three Fermi surfaces, each with its own doping level. '\
        'Read "Dataset A". The doping levels of the three Fermi surfaces "Dataset A" are listed in order as an array; [' + ','.join(str(x) for x in doping_a) + ']. '\
        'Now read "Dataset B". State the doping levels of "Dataset B", in the same order corresponding to each band, as an array of three numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/E3/E3"
    elif size == 2:
        question_name = "Prompts_med/E3/E3"
    elif size == 3:
        question_name = "Prompts_large/E3/E3"
    else:
        question_name = "Prompts_single/E3/E3"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(doping_b), full_name, "_S")
    else:
        write_to_text(str(doping_b), question_name, "_S")

    return question, num





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------- E4 ----- Fermi surface: 3-band nickelate -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_E4(resolution, noise_ratio):

    dynamic_range = 1000
    E0 = 0; E_conv = 0.0001; dE = 0.0001
    E_params = np.array([E0, E_conv, dE])
    xy_bound = 1; bkg = 0

    map_params = np.array([resolution, xy_bound, bkg])

    k_conv_a = 0.025
    coefs_1_a, coefs_2_a, coefs_3_a = get_nickelate_trilayer_coefs_a()
    noise_params_a = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_a = "Plots/E4/E4a"

    k_conv_b = 0.03
    coefs_1_b, coefs_2_b, coefs_3_b = get_nickelate_trilayer_coefs_b()
    noise_params_b = np.concatenate((get_default_map_noise(), noise_ratio), axis = None)
    plot_name_b = "Plots/E4/E4b"

    doping_1_a, doping_2_a, doping_3_a, txt_a = map_nickelate_trilayer_bands(map_params, E_params, noise_params_a, k_conv_a, coefs_1_a, coefs_2_a, coefs_3_a, dynamic_range, plot_name_a)
    doping_1_b, doping_2_b, doping_3_b, txt_b = map_nickelate_trilayer_bands(map_params, E_params, noise_params_b, k_conv_b, coefs_1_b, coefs_2_b, coefs_3_b, dynamic_range, plot_name_b)

    return doping_1_a, doping_2_a, doping_3_a, doping_1_b, doping_2_b, doping_3_b, txt_a, txt_b



def ask_E4(resolution, size, noise_ratio):
    
    doping_1_a, doping_2_a, doping_3_a, doping_1_b, doping_2_b, doping_3_b, txt_a, txt_b = get_E4(int(resolution), noise_ratio)
    doping_a = np.array([doping_1_a, doping_2_a, doping_3_a]); doping_b = np.array([doping_1_b, doping_2_b, doping_3_b])
    doping_a = np.round(doping_a, decimals = 4); doping_b = np.round(doping_b, decimals = 4)
    num = 3

    prompt = 'Two datasets showing ARPES Fermi surfaces in one Brillouin zone are contained. They are labelled "Dataset A" and "Dataset B". '\
        'Doping level is linearly related to the area enclosed by a Fermi surface. '\
        'If the Fermi surface completely encloses the Brillouin zone, it has a doping level of +1. '\
        'If the Fermi surface encloses none of the Brillouin zone, it has a doping level of -1. '\
        'Note that because Brillouin zones have periodic boundaries, the inside and outside of a Brillouin zone may be swapped. '\
        'Here is an example with three Fermi surfaces, each with its own doping level. '\
        'Read "Dataset A". The doping levels of the three Fermi surfaces "Dataset A" are listed in order as an array; [' + ','.join(str(x) for x in doping_a) + ']. '\
        'Now read "Dataset B". State the doping levels of "Dataset B", in the same order corresponding to each band, as an array of three numbers. Print only an array.'
    
    content = "Dataset A\n" + txt_a + "\n\n Dataset B\n" + txt_b

    question = prompt + "\n\n" + content

    if size == 1:
        question_name = "Prompts_small/E4/E4"
    elif size == 2:
        question_name = "Prompts_med/E4/E4"
    elif size == 3:
        question_name = "Prompts_large/E4/E4"
    else:
        question_name = "Prompts_single/E4/E4"

    noise_int = round(100*noise_ratio)
    append_parameters = f"_r{int(resolution)}_n{noise_int}"
    full_name = question_name + append_parameters
    write_to_text(question, full_name, "_Q")

    if size == 0:
        write_to_text(str(doping_b), full_name, "_S")
    else:
        write_to_text(str(doping_b), question_name, "_S")

    return question, num



