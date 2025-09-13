import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import json
from init import write_to_text, get_band_linear_params_d, get_band_quadratic_params_d, create_spectrum, cut_array, get_spectrum_params_a, get_spectrum_params_d, get_spectrum_params_extended_d
from init import fit_FD, fit_L1, fit_L1_EDC, fit_Ln, fit_L1_restrained, find_vF, find_bbE, find_Dirac, find_SC_gap, find_1_phonon, find_2_phonons, find_3_phonons, find_doping_1, find_dopings_2, find_dopings_3
from ask import get_A_titles, get_BCD_titles, get_E_titles
from call_db import get_filename

import matplotlib
matplotlib.use('Agg')





# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------- Scoring and plots ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_spectrum_arrays_full(resolution):

    spectrum_params = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]

    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max, resolution, 1)

    return E_array, k_array



def get_spectrum_arrays(resolution):

    spectrum_params = np.concatenate((resolution, get_spectrum_params_d()), axis = None)
    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    E_array_cut = cut_array(E_array, E_array, mu)

    return E_array_cut, k_array



def get_spectrum_arrays_extended(resolution):

    spectrum_params = np.concatenate((resolution, get_spectrum_params_extended_d()), axis = None)
    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]

    E_max_pad = 1.25*(mu - E_min) + mu; resolution_pad = round(1.25*resolution)
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min, E_max_pad, resolution_pad, 0)
    E_array_cut = cut_array(E_array, E_array, mu)

    return E_array_cut, k_array



def get_spectrum_SC_arrays(resolution):

    spectrum_params = np.concatenate((resolution, get_spectrum_params_a()), axis = None)
    resolution = round(spectrum_params[0]); k_min = spectrum_params[1]; k_max = spectrum_params[2]
    mu = spectrum_params[3]; E_min = spectrum_params[4]; E_max = spectrum_params[5]

    E_min_shifted = E_min - mu
    k_array, E_array, spectrum = create_spectrum(k_min, k_max, resolution, E_min_shifted, -E_min_shifted, round(resolution/2), 0)

    return E_array, k_array



def get_solution_floats(solution_name):

    with open(solution_name, "r") as file:
        content = file.read()
    solution = float(content)

    return solution



def get_solution_arrays(solution_name):

    with open(solution_name, "r") as file:
        content = file.read()
    file.close()
    
    lines = content.strip().split('\n')
    clean_lines = []

    for line in lines:
        if line.strip() and line[0].isdigit():
            parts = line.split(None, 1)

            if len(parts) > 1:
                line = parts[1]
        
        line = line.replace('[', '').replace(']', '').replace('nan', '0')

        if line.strip():
            clean_lines.append(line)
    
    all_data = ' '.join(clean_lines)
    numbers = all_data.split()
    float_numbers = [float(num) for num in numbers]
    solution_array = np.array(float_numbers)
    solution_array = np.ndarray.flatten(solution_array)

    return solution_array



def read_array(array_name):

    with open(array_name, "r") as file:
        content = file.read()
    file.close()
    
    lines = content.strip().split('\n')
    clean_lines = []

    for line in lines:
        """ if line.strip() and line[0].isdigit():
            parts = line.split(None, 1)

            if len(parts) > 1:
                line = parts[1] """
        
        line = line.replace('[', '').replace(']', '').replace('nan', '0')

        if line.strip():
            clean_lines.append(line)
    
    all_data = ' '.join(clean_lines)
    numbers = all_data.split()
    float_numbers = [float(num) for num in numbers]
    read_array = np.array(float_numbers)
    read_array = np.ndarray.flatten(read_array)

    return read_array



def read_spectrum(spectrum_name):

    spectrum_arrays = []

    with open(spectrum_name, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            if line[0] == '{':
                line = ''
            
            line = line.replace('[', '').replace(']', '')
            line = line.replace('null', '0')
            line = line.replace('nan', '0')
            line = line.replace(' nan', ' 0 ')
            line = line.replace(' nan ', ' 0 ')
            line = line.replace('nan', ' 0 ')
            parts = line.strip().split()
            
            if len(parts) > 1 and parts[0].isdigit():
                numeric_data = ' '.join(parts[1:])
            else:
                numeric_data = ' '.join(parts)
            
            numeric_data = numeric_data.replace(',', ' ')
            
            try:
                float_values = [float(val) for val in numeric_data.split() if val.replace('.', '', 1).isdigit()]
                if float_values:
                    spectrum_arrays.append(np.array(float_values))
            except ValueError:
                continue
    
    return np.array(spectrum_arrays)



def get_response_floats(solution_name, response_name):

    with open(solution_name, "r") as file:
        content = file.read()
    solution = float(content)
    responses = []

    try:
        with open(response_name, 'r') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    line = line.strip()

                    if not line:
                        continue
                    
                    line_str = json.dumps(json.loads(line))
                    line_float = float(line_str)
                    responses.append(line_float)
                        
                except Exception as e:
                    pass
    
    except FileNotFoundError:
        pass
    
    except Exception as e:
        pass
    
    # Returns a float, and a list of floats
    return solution, responses



def get_response_arrays(solution_name, response_name):

    # solution_array = []
    response_arrays = []

    # with open(solution_name, "r") as file:
    #     #content = file.read()
    #     for line in file:
    #         if not line.strip():
    #             continue
            
    #         line = line.replace('[', '').replace(']', '')
    #         line = line.replace('null', '0')
    #         line = line.replace('nan', '0')
    #         line = line.replace(' nan', ' 0 ')
    #         line = line.replace(' nan ', ' 0 ')
    #         line = line.replace('nan', ' 0 ')
    #         parts = line.strip().split()
            
    #         if len(parts) > 1 and parts[0].isdigit():
    #             numeric_data = ' '.join(parts[1:])
    #         else:
    #             numeric_data = ' '.join(parts)
            
    #         numeric_data = numeric_data.replace(',', ' ')
            
    #         try:
    #             float_values = [float(val) for val in numeric_data.split() if val.replace('.', '', 1).isdigit()]
    #             if float_values:
    #                 solution_array.append(np.array(float_values))
    #         except ValueError:
    #             continue
    
    # solution_array = np.ndarray.flatten(np.concatenate(solution_array).ravel())
    
    with open(solution_name, "r") as file:
        content = file.read()
    file.close()
    
    lines = content.strip().split('\n')
    clean_lines = []

    for line in lines:
        if line.strip() and line[0].isdigit():
            parts = line.split(None, 1)

            if len(parts) > 1:
                line = parts[1]
        
        line = line.replace('[', '').replace(']', '').replace('nan', '0')

        if line.strip():
            clean_lines.append(line)
    
    all_data = ' '.join(clean_lines)
    numbers = all_data.split()
    float_numbers = [float(num) for num in numbers]
    solution_array = np.array(float_numbers)
    solution_array = np.ndarray.flatten(solution_array)

    # response_arrays = []

    # try:
    #     with open(response_name, 'r') as file:
    #         for line_num, line in enumerate(file, 1):
    #             try:
    #                 line = line.strip()

    #                 if not line:
    #                     continue
                    
    #                 line_str = json.dumps(json.loads(line))
    #                 #line_array = np.fromstring(line_str[1:-1], sep = ', ')
    #                 line_array = np.fromstring(line_str, sep = ', ')
    #                 response_arrays.append(line_array)
                        
    #             except Exception as e:
    #                 pass

    with open(response_name, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            if line[0] == '{':
                line = ''
            
            line = line.replace('[', '').replace(']', '')
            line = line.replace('null', '0')
            line = line.replace('nan', '0')
            line = line.replace(' nan', ' 0 ')
            line = line.replace(' nan ', ' 0 ')
            line = line.replace('nan', ' 0 ')
            parts = line.strip().split()
            
            if len(parts) > 1 and parts[0].isdigit():
                numeric_data = ' '.join(parts[1:])
            else:
                numeric_data = ' '.join(parts)
            
            numeric_data = numeric_data.replace(',', ' ')
            
            try:
                float_values = [float(val) for val in numeric_data.split() if val.replace('.', '', 1).isdigit()]
                if float_values:
                    response_arrays.append(np.array(float_values))
            except ValueError:
                continue
    
    # except FileNotFoundError:
    #     pass
    
    # except Exception as e:
    #     pass
    
    # Returns an array, and a list of arrays
    return solution_array, response_arrays



def get_score_single(ground_truth, response, sigma):
    """ This scores a single response only """

    difference = ground_truth - response
    score_erf = 1 - np.absolute(scipy.special.erf(difference/sigma))
    score_gauss = np.exp(-0.5*((difference/sigma)**2))
    score_lorentz = (sigma**2)/(difference**2 + sigma**2)

    return score_erf, score_gauss, score_lorentz



def get_score_array(ground_truth_array, response_array, sigma):
    """ This scores a single response only """

    if len(response_array) == len(ground_truth_array):
        score_array_erf = np.ones(len(ground_truth_array))
        score_array_gauss = np.ones(len(ground_truth_array))
        score_array_lorentz = np.ones(len(ground_truth_array))

        for i in range(len(score_array_erf)):
            score_erf_single, score_gauss_single, score_lorentz_single = get_score_single(ground_truth_array[i], response_array[i], sigma)
            score_array_erf[i] = score_erf_single
            score_array_gauss[i] = score_gauss_single
            score_array_lorentz[i] = score_lorentz_single
        
        score_erf = np.mean(score_array_erf)
        score_gauss = np.mean(score_array_gauss)
        score_lorentz = np.mean(score_array_lorentz)
        invalid = 0

    else:
        score_erf = 0; score_gauss = 0; score_lorentz = 0
        invalid = 1

    return score_erf, score_gauss, score_lorentz, invalid



def get_score_duple(ground_truth_array, response_array, sigma):
    """ This scores a single response only """

    if len(response_array) == 2:
        score_erf_compiled = np.zeros(2); score_gauss_compiled = np.zeros(2); score_lorentz_compiled = np.zeros(2)

        response_0 = response_array
        score_erf_single_0, score_gauss_single_0, score_lorentz_single_0, invalid_0 = get_score_array(ground_truth_array, response_0, sigma)
        score_erf_compiled[0] = score_erf_single_0; score_gauss_compiled[0] = score_gauss_single_0; score_lorentz_compiled[0] = score_lorentz_single_0

        response_1 = np.array([response_array[1], response_array[0]])
        score_erf_single_1, score_gauss_single_1, score_lorentz_single_1, invalid_0 = get_score_array(ground_truth_array, response_1, sigma)
        score_erf_compiled[1] = score_erf_single_1; score_gauss_compiled[1] = score_gauss_single_1; score_lorentz_compiled[1] = score_lorentz_single_1

        score_erf = np.max(score_erf_compiled)
        score_gauss = np.max(score_gauss_compiled)
        score_lorentz = np.max(score_lorentz_compiled)
        invalid = 0

    else:
        score_erf = 0; score_gauss = 0; score_lorentz = 0
        invalid = 1

    return score_erf, score_gauss, score_lorentz, invalid



def get_score_triple(ground_truth_array, response_array, sigma):
    """ This scores a single response only """

    if len(response_array) == 3:
        score_erf_compiled = np.zeros(6); score_gauss_compiled = np.zeros(6); score_lorentz_compiled = np.zeros(6)

        response_0 = response_array
        score_erf_single_0, score_gauss_single_0, score_lorentz_single_0, invalid_0 = get_score_array(ground_truth_array, response_0, sigma)
        score_erf_compiled[0] = score_erf_single_0; score_gauss_compiled[0] = score_gauss_single_0; score_lorentz_compiled[0] = score_lorentz_single_0

        response_1 = np.array([response_array[2], response_array[0], response_array[1]])
        score_erf_single_1, score_gauss_single_1, score_lorentz_single_1, invalid_0 = get_score_array(ground_truth_array, response_1, sigma)
        score_erf_compiled[1] = score_erf_single_1; score_gauss_compiled[1] = score_gauss_single_1; score_lorentz_compiled[1] = score_lorentz_single_1

        response_2 = np.array([response_array[1], response_array[2], response_array[0]])
        score_erf_single_2, score_gauss_single_2, score_lorentz_single_2, invalid_0 = get_score_array(ground_truth_array, response_2, sigma)
        score_erf_compiled[2] = score_erf_single_2; score_gauss_compiled[2] = score_gauss_single_2; score_lorentz_compiled[2] = score_lorentz_single_2

        response_3 = np.array([response_array[2], response_array[1], response_array[0]])
        score_erf_single_3, score_gauss_single_3, score_lorentz_single_3, invalid_0 = get_score_array(ground_truth_array, response_3, sigma)
        score_erf_compiled[3] = score_erf_single_3; score_gauss_compiled[3] = score_gauss_single_3; score_lorentz_compiled[3] = score_lorentz_single_3

        response_4 = np.array([response_array[0], response_array[2], response_array[1]])
        score_erf_single_4, score_gauss_single_4, score_lorentz_single_4, invalid_0 = get_score_array(ground_truth_array, response_4, sigma)
        score_erf_compiled[4] = score_erf_single_4; score_gauss_compiled[4] = score_gauss_single_4; score_lorentz_compiled[4] = score_lorentz_single_4

        response_5 = np.array([response_array[1], response_array[0], response_array[2]])
        score_erf_single_5, score_gauss_single_5, score_lorentz_single_5, invalid_0 = get_score_array(ground_truth_array, response_5, sigma)
        score_erf_compiled[5] = score_erf_single_5; score_gauss_compiled[5] = score_gauss_single_5; score_lorentz_compiled[5] = score_lorentz_single_5

        score_erf = np.max(score_erf_compiled)
        score_gauss = np.max(score_gauss_compiled)
        score_lorentz = np.max(score_lorentz_compiled)
        invalid = 0

    else:
        score_erf = 0; score_gauss = 0; score_lorentz = 0
        invalid = 1

    return score_erf, score_gauss, score_lorentz, invalid



def score_code_floats(solution, response, sigma):
    score_erf, score_gauss, score_lorentz = get_score_single(solution, response, sigma)
    return score_erf, score_gauss, score_lorentz



def score_code_arrays(solution, response, sigma):

    if len(solution) == 2:
        score_erf, score_gauss, score_lorentz, invalid = get_score_duple(solution, response, sigma)
    elif len(solution) == 3:
        score_erf, score_gauss, score_lorentz, invalid = get_score_triple(solution, response, sigma)
    else:
        score_erf, score_gauss, score_lorentz, invalid = get_score_array(solution, response, sigma)

    return score_erf, score_gauss, score_lorentz



def score_response_floats(solution_name, response_name, sigma):

    solution, responses = get_response_floats(solution_name, response_name)
    num = len(responses)

    if num > 0:
        score_erf_all = np.zeros(num); score_gauss_all = np.zeros(num); score_lorentz_all = np.zeros(num)

        for i in range(num):
            score_erf, score_gauss, score_lorentz = get_score_single(solution, responses[i], sigma)
            score_erf_all[i] = score_erf; score_gauss_all[i] = score_gauss; score_lorentz_all[i] = score_lorentz
        
        score_erf = np.mean(score_erf_all); score_gauss = np.mean(score_gauss_all); score_lorentz = np.mean(score_lorentz_all)

        if num > 1:
            std_erf = np.std(score_erf_all, ddof = 1); std_gauss = np.std(score_gauss_all, ddof = 1); std_lorentz = np.std(score_lorentz_all, ddof = 1)
        
        else:
            std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
    else:
        score_erf = np.nan; score_gauss = np.nan; score_lorentz = np.nan
        std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan

    return score_erf, score_gauss, score_lorentz, std_erf, std_gauss, std_lorentz



def score_response_floats_all(solution_name, response_name, sigma):

    solution, responses = get_response_floats(solution_name, response_name)
    num = len(responses)

    if num > 0:
        score_erf_all = np.zeros(num); score_gauss_all = np.zeros(num); score_lorentz_all = np.zeros(num)

        for i in range(num):
            score_erf, score_gauss, score_lorentz = get_score_single(solution, responses[i], sigma)
            score_erf_all[i] = score_erf; score_gauss_all[i] = score_gauss; score_lorentz_all[i] = score_lorentz
        
        score_erf = np.mean(score_erf_all); score_gauss = np.mean(score_gauss_all); score_lorentz = np.mean(score_lorentz_all)

        if num > 1:
            std_erf = np.std(score_erf_all, ddof = 1); std_gauss = np.std(score_gauss_all, ddof = 1); std_lorentz = np.std(score_lorentz_all, ddof = 1)
        
        else:
            std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
    else:
        score_erf = np.nan; score_gauss = np.nan; score_lorentz = np.nan
        score_erf_all = np.array([np.nan]); score_gauss_all = np.array([np.nan]); score_lorentz_all = np.array([np.nan])
        std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan

    return score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz



def score_response_arrays(solution_name, response_name, sigma):

    solution_array, response_arrays = get_response_arrays(solution_name, response_name)
    num = len(response_arrays)
    invalid_all = 0

    if num > 0:
        score_erf_all = np.zeros(num); score_gauss_all = np.zeros(num); score_lorentz_all = np.zeros(num)

        for i in range(num):
            if len(solution_array) == 2:
                score_erf, score_gauss, score_lorentz, invalid = get_score_duple(solution_array, response_arrays[i], sigma)
            elif len(solution_array) == 3:
                score_erf, score_gauss, score_lorentz, invalid = get_score_triple(solution_array, response_arrays[i], sigma)
            else:
                score_erf, score_gauss, score_lorentz, invalid = get_score_array(solution_array, response_arrays[i], sigma)
            
            score_erf_all[i] = score_erf; score_gauss_all[i] = score_gauss; score_lorentz_all[i] = score_lorentz
            invalid_all += invalid/num
        
        score_erf = np.mean(score_erf_all); score_gauss = np.mean(score_gauss_all); score_lorentz = np.mean(score_lorentz_all)
        score_erf *= (1 - invalid_all); score_gauss *= (1 - invalid_all); score_lorentz *= (1 - invalid_all); 

        if num > 1:
            std_erf = np.std(score_erf_all, ddof = 1); std_gauss = np.std(score_gauss_all, ddof = 1); std_lorentz = np.std(score_lorentz_all, ddof = 1)
        else:
            std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
    else:
        score_erf = np.nan; score_gauss = np.nan; score_lorentz = np.nan
        std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
        invalid_all = 1

    return score_erf, score_gauss, score_lorentz, std_erf, std_gauss, std_lorentz, invalid_all



def score_response_arrays_all(solution_name, response_name, sigma):

    solution_array, response_arrays = get_response_arrays(solution_name, response_name)
    num = len(response_arrays)
    invalid_all = 0

    if num > 0:
        score_erf_all = np.zeros(num); score_gauss_all = np.zeros(num); score_lorentz_all = np.zeros(num)

        for i in range(num):
            if len(solution_array) == 2:
                score_erf, score_gauss, score_lorentz, invalid = get_score_duple(solution_array, response_arrays[i], sigma)
            elif len(solution_array) == 3:
                score_erf, score_gauss, score_lorentz, invalid = get_score_triple(solution_array, response_arrays[i], sigma)
            else:
                score_erf, score_gauss, score_lorentz, invalid = get_score_array(solution_array, response_arrays[i], sigma)
            
            score_erf_all[i] = score_erf; score_gauss_all[i] = score_gauss; score_lorentz_all[i] = score_lorentz
            invalid_all += invalid/num
        
        score_erf = np.mean(score_erf_all); score_gauss = np.mean(score_gauss_all); score_lorentz = np.mean(score_lorentz_all)
        score_erf *= (1 - invalid_all); score_gauss *= (1 - invalid_all); score_lorentz *= (1 - invalid_all); 

        if num > 1:
            std_erf = np.std(score_erf_all, ddof = 1); std_gauss = np.std(score_gauss_all, ddof = 1); std_lorentz = np.std(score_lorentz_all, ddof = 1)
        else:
            std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
    else:
        score_erf = np.nan; score_gauss = np.nan; score_lorentz = np.nan
        score_erf_all = np.array([np.nan]); score_gauss_all = np.array([np.nan]); score_lorentz_all = np.array([np.nan])
        std_erf = np.nan; std_gauss = np.nan; std_lorentz = np.nan
        invalid_all = 1

    return score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid_all



def plot_histogram_general_single(response_array, answer, sigma, width_multiple, axis_label, file_name):

    response_array = np.array(response_array, dtype = float)
    clean_data = response_array[~np.isnan(response_array)]

    n_bins = 100
    bin_num = np.linspace(answer - sigma*width_multiple, answer + sigma*width_multiple, n_bins)
    weights = np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(answer - sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(answer, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(answer + sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(answer - sigma*width_multiple, answer + sigma*width_multiple)
    plt.ylabel(r'Response frequency')
    plt.xlabel(axis_label)
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_histogram_E_single(response_array, E_array_cut, E_answer, E_conv, file_name):

    response_array = np.array(response_array, dtype = float)
    clean_data = response_array[~np.isnan(response_array)]

    n_bins = 100
    bin_num = np.linspace(E_array_cut.min(), E_array_cut.max(), n_bins)
    weights = np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(E_answer - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(min(E_array_cut), max(E_array_cut))
    plt.ylabel(r'Response frequency')
    plt.xlabel(r'Energy $\omega$ (eV)')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_histogram_E_double(response_array_full, E_array_cut, E_answer_1, E_answer_2, E_conv, file_name):

    response_array = np.ndarray.flatten(np.concatenate(response_array_full).ravel())
    response_array = np.array(response_array, dtype = float)
    clean_data = response_array[~np.isnan(response_array)]

    n_bins = 100
    bin_num = np.linspace(E_array_cut.min(), E_array_cut.max(), n_bins)
    weights = 2*np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(E_answer_1 - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer_1, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer_1 + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(E_answer_2 - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer_2, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer_2 + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(min(E_array_cut), max(E_array_cut))
    plt.ylabel('Response frequency')
    plt.xlabel(r'Energy $\omega$ (eV)')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_histogram_E_triple(response_array_full, E_array_cut, E_answer_1, E_answer_2, E_answer_3, E_conv, file_name):

    response_array = np.ndarray.flatten(np.concatenate(response_array_full).ravel())
    response_array = np.array(response_array, dtype = float)
    clean_data = response_array[~np.isnan(response_array)]
    
    n_bins = 100
    bin_num = np.linspace(E_array_cut.min(), E_array_cut.max(), n_bins)
    weights = 3*np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(E_answer_1 - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer_1, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer_1 + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(E_answer_2 - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer_2, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer_2 + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(E_answer_3 - E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(E_answer_3, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(E_answer_3 + E_conv, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(min(E_array_cut), max(E_array_cut))
    plt.ylabel(r'Response frequency')
    plt.xlabel(r'Energy $\omega$ (eV)')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_dispersion_E(response_arrays, dispersion_array, E_array_cut, k_array, k_conv, file_name):

    plt.plot(dispersion_array, E_array_cut, 'r', linewidth = 3)
    plt.plot(dispersion_array - k_conv, E_array_cut, 'r--', linewidth = 1)
    plt.plot(dispersion_array + k_conv, E_array_cut, 'r--', linewidth = 1)

    num = len(response_arrays)
    if num > 0:
        for i in range(len(response_arrays)):
            response_array_single = response_arrays[i]
            if len(response_array_single) == len(dispersion_array):
                plt.plot(response_array_single, E_array_cut, 'k', linewidth = 0.5)
    
    plt.xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    plt.ylabel(r'Energy $\omega$ (eV)')
    plt.ylim(np.min(E_array_cut), np.max(E_array_cut))
    plt.xlim(np.min(k_array), np.max(k_array))
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_dispersion_k(response_arrays, Ek_array_cut, E_array_cut, k_array, E_conv, file_name):

    plt.plot(k_array, Ek_array_cut, 'r', linewidth = 3)
    plt.plot(k_array, Ek_array_cut - E_conv, 'r--', linewidth = 1)
    plt.plot(k_array, Ek_array_cut + E_conv, 'r--', linewidth = 1)

    num = len(response_arrays)
    if num > 0:
        for i in range(len(response_arrays)):
            response_array_single = response_arrays[i]
            if len(response_array_single) == len(Ek_array_cut):
                plt.plot(k_array, response_array_single, 'k', linewidth = 0.5)
    
    plt.xlabel(r'Momentum $k$ ($\AA^{-1}$)')
    plt.ylabel(r'Energy $\omega$ (eV)')
    plt.ylim(np.min(E_array_cut), np.max(E_array_cut))
    plt.xlim(np.min(k_array), np.max(k_array))
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_linewidth(response_arrays, linewidth_array, E_array_cut, k_array, k_conv, file_name):

    plt.plot(E_array_cut, linewidth_array, 'r', linewidth = 3)
    plt.plot(E_array_cut, linewidth_array - k_conv, 'r--', linewidth = 1)
    plt.plot(E_array_cut, linewidth_array + k_conv, 'r--', linewidth = 1)
    finite_arr = linewidth_array[np.isfinite(linewidth_array)]

    num = len(response_arrays)
    if num > 0:
        for i in range(len(response_arrays)):
            response_array_single = response_arrays[i]
            if len(response_array_single) == len(linewidth_array):
                plt.plot(E_array_cut, response_array_single, 'k', linewidth = 0.5)
    
    plt.xlabel(r'Energy $\omega$ (eV)')
    plt.ylabel(r'FWHM ($\AA^{-1}$)')
    plt.xlim(np.min(E_array_cut), np.max(E_array_cut))
    plt.ylim(0, 1.1*(np.max(finite_arr) +  k_conv))
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_doping_single(response_array, doping_answer, doping_sigma, file_name):

    response_array = np.array(response_array, dtype = float)
    clean_data = response_array[~np.isnan(response_array)]

    n_bins = 100
    bin_num = np.linspace(-1, 1, n_bins)
    weights = np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(doping_answer - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(-1, 1)
    plt.ylabel(r'Response frequency')
    plt.xlabel(r'Doping level')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_doping_double(response_array_full, doping_answer_1, doping_answer_2, doping_sigma, file_name):

    try:
        response_array = np.ndarray.flatten(np.concatenate(response_array_full).ravel())
        response_array = np.array(response_array, dtype = float)
        clean_data = response_array[~np.isnan(response_array)]
    except ValueError:
        response_array = []

    n_bins = 100
    bin_num = np.linspace(-1, 1, n_bins)
    weights = np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(doping_answer_1 - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer_1, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer_1 + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(doping_answer_2 - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer_2, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer_2 + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(-1, 1)
    plt.ylabel(r'Response frequency')
    plt.xlabel(r'Doping level')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return



def plot_doping_triple(response_array_full, doping_answer_1, doping_answer_2, doping_answer_3, doping_sigma, file_name):

    try:
        response_array = np.ndarray.flatten(np.concatenate(response_array_full).ravel())
        response_array = np.array(response_array, dtype = float)
        clean_data = response_array[~np.isnan(response_array)]
    except ValueError:
        response_array = []

    n_bins = 100
    bin_num = np.linspace(-1, 1, n_bins)
    weights = np.ones_like(response_array)/len(response_array)
    plt.hist(response_array, bins = bin_num, weights = weights, color = "skyblue", ec = "skyblue", stacked = True, density = False)

    plt.axvline(doping_answer_1 - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer_1, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer_1 + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(doping_answer_2 - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer_2, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer_2 + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.axvline(doping_answer_3 - doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)
    plt.axvline(doping_answer_3, color = 'r', linestyle = 'solid', linewidth = 1)
    plt.axvline(doping_answer_3 + doping_sigma, color = 'r', linestyle = 'dotted', linewidth = 1)

    plt.xlim(-1, 1)
    plt.ylabel(r'Response frequency')
    plt.xlabel(r'Doping level')
    #plt.title(file_name)

    full_path = f'/workspaces/physics-benchmark/client/Evaluation_plots/{file_name}'
    plt.savefig(full_path)
    plt.show()
    plt.close()
    return





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------- Plot all questions ---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def plot_model_responses(model, size):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01
    E_factor = 1; k_factor = 1; doping_factor = 3
    E_conv *= E_factor; k_conv *= k_factor; doping_sigma *= doping_factor

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    if size == 1:
        resolution_A1 = 75
        resolution_BCD = 100
        resolution_E1 = 80
        resolution_E234 = 110
    elif size == 2:
        resolution_A1 = 125
        resolution_BCD = 200
        resolution_E1 = 150
        resolution_E234 = 220
    else:
        resolution_A1 = 250
        resolution_BCD = 300
        resolution_E1 = 250
        resolution_E234 = 350

    if size == 1:
        response_prefix = "Responses_small/"
        response_suffix = ""
    elif size == 2:
        response_prefix = "Responses_med/"
        response_suffix = "_med"
    else:
        response_prefix = "Responses_large/"
        response_suffix = "_large"

    for Q_num in range(1, 6):
        # Questions A1

        if size == 1:
            solution_name = "Prompts_small/A1/A1_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/A1/A1_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/A1/A1_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array, k_array = get_spectrum_arrays_full(resolution_A1)
        plot_histogram_E_single(responses, E_array, solution, E_conv, file_name)
    
    for Q_num in range(6, 11):
        # Questions B1

        if size == 1:
            solution_name = "Prompts_small/B1/B1_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B1/B1_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B1/B1_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(11, 16):
        # Questions B1 (vF)

        if size == 1:
            solution_name = "Prompts_small/B1/B1_vF_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B1/B1_vF_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B1/B1_vF_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        noise_level = int((Q_num - 6)*10)
        spectrum_name = f"Spectra/B1_r{int(resolution_BCD)}_n{noise_level}"
    
        solution = get_solution_floats(solution_name)
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        axis_label = r'Fermi velocity (eV$\cdot\AA$)'
        plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
    
    for Q_num in range(16, 21):
        # Questions B2

        if size == 1:
            solution_name = "Prompts_small/B2/B2_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B2/B2_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B2/B2_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(21, 26):
        # Questions B2 (vF)

        if size == 1:
            solution_name = "Prompts_small/B2/B2_vF_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B2/B2_vF_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B2/B2_vF_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"

        noise_level = int((Q_num - 16)*10)
        spectrum_name = f"Spectra/B2_r{int(resolution_BCD)}_n{noise_level}"
    
        solution = get_solution_floats(solution_name)
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_quadratic*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
    
    for Q_num in range(26, 31):
        # Questions B3

        if size == 1:
            solution_name = "Prompts_small/B3/B3_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B3/B3_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B3/B3_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(31, 36):
        # Questions B3 (vF)

        if size == 1:
            solution_name = "Prompts_small/B3/B3_vF_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B3/B3_vF_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B3/B3_vF_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"

        noise_level = int((Q_num - 26)*10)
        spectrum_name = f"Spectra/B3_r{int(resolution_BCD)}_n{noise_level}"
    
        solution = get_solution_floats(solution_name)
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
    
    for Q_num in range(36, 41):
        # Questions B4

        if size == 1:
            solution_name = "Prompts_small/B4/B4_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B4/B4_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B4/B4_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_dispersion_k(response_arrays, solution_array, E_array_cut, k_array, E_conv, file_name)
    
    for Q_num in range(41, 46):
        # Questions B4 (bbE)

        if size == 1:
            solution_name = "Prompts_small/B4/B4_bbE_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B4/B4_bbE_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B4/B4_bbE_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
    
    for Q_num in range(46, 51):
        # Questions B5

        if size == 1:
            solution_name = "Prompts_small/B5/B5_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B5/B5_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B5/B5_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
    
    for Q_num in range(51, 56):
        # Questions B6

        if size == 1:
            solution_name = "Prompts_small/B6/B6_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/B6/B6_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/B6/B6_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_SC_arrays(resolution_BCD)
        E_lims = np.array([0, np.max(max(E_array_cut))])
        plot_histogram_E_single(responses, E_lims, solution, E_conv, file_name)
    
    for Q_num in range(56, 61):
        # Questions C1

        if size == 1:
            solution_name = "Prompts_small/C1/C1_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/C1/C1_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/C1/C1_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(61, 66):
        # Questions C2

        if size == 1:
            solution_name = "Prompts_small/C2/C2_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/C2/C2_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/C2/C2_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(66, 71):
        # Questions C3

        if size == 1:
            solution_name = "Prompts_small/C3/C3_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/C3/C3_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/C3/C3_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(71, 76):
        # Questions C4

        if size == 1:
            solution_name = "Prompts_small/C4/C4_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/C4/C4_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/C4/C4_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(76, 81):
        # Questions C5

        if size == 1:
            solution_name = "Prompts_small/C5/C5_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/C5/C5_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/C5/C5_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays(resolution_BCD)
        plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
    
    for Q_num in range(81, 106):
        # Questions D1

        if size == 1:
            solution_name = "Prompts_small/D1/D1_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/D1/D1_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/D1/D1_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays_extended(resolution_BCD)
        plot_histogram_E_single(responses, E_array_cut, solution, E_conv*3, file_name)
    
    for Q_num in range(106, 111):
        # Questions D2

        if size == 1:
            solution_name = "Prompts_small/D2/D2_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/D2/D2_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/D2/D2_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays_extended(resolution_BCD)
        plot_histogram_E_double(response_arrays, E_array_cut, solution_array[0], solution_array[1], E_conv*3, file_name)
    
    for Q_num in range(111, 116):
        # Questions D3

        if size == 1:
            solution_name = "Prompts_small/D3/D3_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/D3/D3_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/D3/D3_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        E_array_cut, k_array = get_spectrum_arrays_extended(resolution_BCD)
        plot_histogram_E_triple(response_arrays, E_array_cut, solution_array[0], solution_array[1], solution_array[2], E_conv*3, file_name)
    
    for Q_num in range(116, 121):
        # Questions E1

        if size == 1:
            solution_name = "Prompts_small/E1/E1_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/E1/E1_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/E1/E1_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        plot_doping_single(responses, solution, doping_sigma, file_name)
    
    for Q_num in range(121, 126):
        # Questions E2

        if size == 1:
            solution_name = "Prompts_small/E2/E2_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/E2/E2_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/E2/E2_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        plot_doping_double(response_arrays, solution_array[0], solution_array[1], doping_sigma, file_name)
    
    for Q_num in range(126, 131):
        # Questions E3

        if size == 1:
            solution_name = "Prompts_small/E3/E3_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/E3/E3_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/E3/E3_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        plot_doping_triple(response_arrays, solution_array[0], solution_array[1], solution_array[2], doping_sigma, file_name)
    
    for Q_num in range(131, 136):
        # Questions E4

        if size == 1:
            solution_name = "Prompts_small/E4/E4_S.txt"
            file_name = model + "_small" + f"-Q{int(Q_num)}.png"
        elif size == 2:
            solution_name = "Prompts_med/E4/E4_S.txt"
            file_name = model + "_med" + f"-Q{int(Q_num)}{response_suffix}.png"
        else:
            solution_name = "Prompts_large/E4/E4_S.txt"
            file_name = model + "_large" + f"-Q{int(Q_num)}{response_suffix}.png"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        plot_doping_triple(response_arrays, solution_array[0], solution_array[1], solution_array[2], doping_sigma, file_name)

    return





# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------- Score all questions ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def score_model_responses(model, size, model_name):

    all_scores_erf = np.zeros(135); all_scores_gauss = np.zeros(135); all_scores_lorentz = np.zeros(135); all_err_n = np.zeros(135); all_err_p = np.zeros(135)
    all_std_erf = np.zeros(135); all_std_gauss = np.zeros(135); all_std_lorentz = np.zeros(135)
    all_scores_erf.fill(np.nan); all_scores_gauss.fill(np.nan); all_scores_lorentz.fill(np.nan); all_err_n.fill(np.nan); all_err_p.fill(np.nan)
    all_std_erf.fill(np.nan); all_std_gauss.fill(np.nan); all_std_lorentz.fill(np.nan)

    T1_scores = np.zeros(5); T2_scores = np.zeros(5); T3_scores = np.zeros(5)
    
    all_invalid = np.zeros(135)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    if size == 1:
        resolution_A1 = 75
        resolution_BCD = 100
        resolution_E1 = 80
        resolution_E234 = 110
    elif size == 2:
        resolution_A1 = 125
        resolution_BCD = 200
        resolution_E1 = 150
        resolution_E234 = 220
    else:
        resolution_A1 = 250
        resolution_BCD = 300
        resolution_E1 = 250
        resolution_E234 = 350

    if size == 1:
        response_prefix = "Responses_small/"
        response_suffix = ""
    elif size == 2:
        response_prefix = "Responses_med/"
        response_suffix = "_med"
    else:
        response_prefix = "Responses_large/"
        response_suffix = "_large"

    for Q_num in range(1, 6):
        # Questions A1

        if size == 1:
            solution_name = "Prompts_small/A1/A1_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/A1/A1_S.txt"
        else:
            solution_name = "Prompts_large/A1/A1_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(6, 11):
        # Questions B1

        if size == 1:
            solution_name = "Prompts_small/B1/B1_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B1/B1_S.txt"
        else:
            solution_name = "Prompts_large/B1/B1_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(11, 16):
        # Questions B1 (vF)

        if size == 1:
            solution_name = "Prompts_small/B1/B1_vF_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B1/B1_vF_S.txt"
        else:
            solution_name = "Prompts_large/B1/B1_vF_S.txt"

        noise_level = int((Q_num - 11)*10)
        spectrum_name = f"Spectra/B1_r{int(resolution_BCD)}_n{noise_level}"
        
        solution = get_solution_floats(solution_name)
       
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, vF_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(16, 21):
        # Questions B2

        if size == 1:
            solution_name = "Prompts_small/B2/B2_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B2/B2_S.txt"
        else:
            solution_name = "Prompts_large/B2/B2_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(21, 26):
        # Questions B2 (vF)

        if size == 1:
            solution_name = "Prompts_small/B2/B2_vF_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B2/B2_vF_S.txt"
        else:
            solution_name = "Prompts_large/B2/B2_vF_S.txt"

        noise_level = int((Q_num - 21)*10)
        spectrum_name = f"Spectra/B2_r{int(resolution_BCD)}_n{noise_level}"
    
        solution = get_solution_floats(solution_name)
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_quadratic*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, vF_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(26, 31):
        # Questions B3

        if size == 1:
            solution_name = "Prompts_small/B3/B3_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B3/B3_S.txt"
        else:
            solution_name = "Prompts_large/B3/B3_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(31, 36):
        # Questions B3 (vF)

        if size == 1:
            solution_name = "Prompts_small/B3/B3_vF_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B3/B3_vF_S.txt"
        else:
            solution_name = "Prompts_large/B3/B3_vF_S.txt"

        noise_level = int((Q_num - 31)*10)
        spectrum_name = f"Spectra/B3_r{int(resolution_BCD)}_n{noise_level}"
    
        solution = get_solution_floats(solution_name)
        spectrum = read_spectrum(spectrum_name + "_sp.txt")
        k_array = read_array(spectrum_name + "_k.txt")[1:]
        E_array = read_array(spectrum_name + "_E.txt")
        
        vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, vF_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(36, 41):
        # Questions B4

        if size == 1:
            solution_name = "Prompts_small/B4/B4_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B4/B4_S.txt"
        else:
            solution_name = "Prompts_large/B4/B4_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(41, 46):
        # Questions B4 (bbE)

        if size == 1:
            solution_name = "Prompts_small/B4/B4_bbE_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B4/B4_bbE_S.txt"
        else:
            solution_name = "Prompts_large/B4/B4_bbE_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(46, 51):
        # Questions B5

        if size == 1:
            solution_name = "Prompts_small/B5/B5_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B5/B5_S.txt"
        else:
            solution_name = "Prompts_large/B5/B5_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(51, 56):
        # Questions B6

        if size == 1:
            solution_name = "Prompts_small/B6/B6_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/B6/B6_S.txt"
        else:
            solution_name = "Prompts_large/B6/B6_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(56, 61):
        # Questions C1

        if size == 1:
            solution_name = "Prompts_small/C1/C1_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/C1/C1_S.txt"
        else:
            solution_name = "Prompts_large/C1/C1_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(61, 66):
        # Questions C2

        if size == 1:
            solution_name = "Prompts_small/C2/C2_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/C2/C2_S.txt"
        else:
            solution_name = "Prompts_large/C2/C2_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(66, 71):
        # Questions C3

        if size == 1:
            solution_name = "Prompts_small/C3/C3_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/C3/C3_S.txt"
        else:
            solution_name = "Prompts_large/C3/C3_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(71, 76):
        # Questions C4

        if size == 1:
            solution_name = "Prompts_small/C4/C4_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/C4/C4_S.txt"
        else:
            solution_name = "Prompts_large/C4/C4_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(76, 81):
        # Questions C5

        if size == 1:
            solution_name = "Prompts_small/C5/C5_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/C5/C5_S.txt"
        else:
            solution_name = "Prompts_large/C5/C5_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T2_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(81, 106):
        # Questions D1

        if size == 1:
            solution_name = "Prompts_small/D1/D1_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/D1/D1_S.txt"
        else:
            solution_name = "Prompts_large/D1/D1_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv*3)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(106, 111):
        # Questions D2

        if size == 1:
            solution_name = "Prompts_small/D2/D2_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/D2/D2_S.txt"
        else:
            solution_name = "Prompts_large/D2/D2_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, E_conv*3)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(111, 116):
        # Questions D3

        if size == 1:
            solution_name = "Prompts_small/D3/D3_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/D3/D3_S.txt"
        else:
            solution_name = "Prompts_large/D3/D3_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, E_conv*3)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T1_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(116, 121):
        # Questions E1

        if size == 1:
            solution_name = "Prompts_small/E1/E1_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/E1/E1_S.txt"
        else:
            solution_name = "Prompts_large/E1/E1_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, doping_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = 0
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(121, 126):
        # Questions E2

        if size == 1:
            solution_name = "Prompts_small/E2/E2_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/E2/E2_S.txt"
        else:
            solution_name = "Prompts_large/E2/E2_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, doping_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(126, 131):
        # Questions E3

        if size == 1:
            solution_name = "Prompts_small/E3/E3_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/E3/E3_S.txt"
        else:
            solution_name = "Prompts_large/E3/E3_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, doping_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score
    
    for Q_num in range(131, 136):
        # Questions E4

        if size == 1:
            solution_name = "Prompts_small/E4/E4_S.txt"
        elif size == 2:
            solution_name = "Prompts_med/E4/E4_S.txt"
        else:
            solution_name = "Prompts_large/E4/E4_S.txt"
        
        response_name = response_prefix + model + f"-Q{int(Q_num)}{response_suffix}.jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, doping_sigma)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        all_std_erf[int(Q_num - 1)] = std_erf; all_std_gauss[int(Q_num - 1)] = std_gauss; all_std_lorentz[int(Q_num - 1)] = std_lorentz
        all_invalid[int(Q_num - 1)] = invalid
        T3_scores[(Q_num - 1)%5] += score_lorentz
        err_score = np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_n[int(Q_num - 1)] = score_lorentz - err_score; all_err_p[int(Q_num - 1)] = score_lorentz + err_score

    results = np.concatenate((all_scores_erf, all_scores_gauss, all_scores_lorentz, all_std_erf, all_std_gauss, all_std_lorentz), axis = 0)
    eval_name = "Evaluation/" + model

    if size == 1:
        write_to_text(str(results), eval_name, "_small")
        resolution = r" (low res)"
    elif size == 2:
        write_to_text(str(results), eval_name, "_med")
        resolution = r" (med res)"
    else:
        write_to_text(str(results), eval_name, "_large")
        resolution = r" (high res)"
    
    question_names = [' ', ' ', r'Fermi Level (A1)', ' ', ' ']
    question_names.extend([' ', ' ', r'Linear (B1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Linear $v_F$ (B1)', '', ''])
    question_names.extend([' ', ' ', r'Quadratic (B2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Quadratic $v_F$ (B2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superstructure (B3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superstructure $v_F$ (B3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Band bottom (B4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Band bottom energy (B4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Dirac cone energy (B5)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superconducting gap size (B6)', ' ', ' '])
    question_names.extend([' ', ' ', r'Impurity scattering (C1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Marginal Fermi liquid, MFL (C2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Fermi liquid, FL (C3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon + MFL (C4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon + FL (C5)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 0.5$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 0.75$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 1$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 2$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 5$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Two phonons (D2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Three phonons (D3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Cuprate, single layer (E1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Cuprate, bilayer (E2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Sr$_2$RuO$_4$ (E3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Nickelate, trilayer (E4)', ' ', ' '])
    
    model_resolution = model + resolution
    question_numbers = np.arange(1, 136)

    # ----------------------------------------------------------------- Plot main scores -----------------------------------------------------------------

    fig = plt.figure('Parallel', figsize = (6, 8), dpi = 500)
    plt.subplots_adjust(left = 0.3)

    plt.plot(all_scores_lorentz[0:5], question_numbers[0:5], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[5:10], question_numbers[5:10], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[10:15], question_numbers[10:15], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[15:20], question_numbers[15:20], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[20:25], question_numbers[20:25], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[25:30], question_numbers[25:30], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[30:35], question_numbers[30:35], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[35:40], question_numbers[35:40], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[40:45], question_numbers[40:45], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[45:50], question_numbers[45:50], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[50:55], question_numbers[50:55], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[55:60], question_numbers[55:60], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[60:65], question_numbers[60:65], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[65:70], question_numbers[65:70], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[70:75], question_numbers[70:75], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[75:80], question_numbers[75:80], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[80:85], question_numbers[80:85], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[85:90], question_numbers[85:90], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[90:95], question_numbers[90:95], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[95:100], question_numbers[95:100], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[100:105], question_numbers[100:105], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[105:110], question_numbers[105:110], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[110:115], question_numbers[110:115], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[115:120], question_numbers[115:120], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[120:125], question_numbers[120:125], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[125:130], question_numbers[125:130], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[130:135], question_numbers[130:135], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.fill_betweenx(question_numbers[0:5], all_err_n[0:5], all_err_p[0:5], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[5:10], all_err_n[5:10], all_err_p[5:10], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[10:15], all_err_n[10:15], all_err_p[10:15], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[15:20], all_err_n[15:20], all_err_p[15:20], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[20:25], all_err_n[20:25], all_err_p[20:25], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[25:30], all_err_n[25:30], all_err_p[25:30], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[30:35], all_err_n[30:35], all_err_p[30:35], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[35:40], all_err_n[35:40], all_err_p[35:40], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[40:45], all_err_n[40:45], all_err_p[40:45], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[45:50], all_err_n[45:50], all_err_p[45:50], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[50:55], all_err_n[50:55], all_err_p[50:55], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[55:60], all_err_n[55:60], all_err_p[55:60], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[60:65], all_err_n[60:65], all_err_p[60:65], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[65:70], all_err_n[65:70], all_err_p[65:70], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[70:75], all_err_n[70:75], all_err_p[70:75], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[75:80], all_err_n[75:80], all_err_p[75:80], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[80:85], all_err_n[80:85], all_err_p[80:85], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[85:90], all_err_n[85:90], all_err_p[85:90], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[90:95], all_err_n[90:95], all_err_p[90:95], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[95:100], all_err_n[95:100], all_err_p[95:100], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[100:105], all_err_n[100:105], all_err_p[100:105], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[105:110], all_err_n[105:110], all_err_p[105:110], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[110:115], all_err_n[110:115], all_err_p[110:115], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[115:120], all_err_n[115:120], all_err_p[115:120], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[120:125], all_err_n[120:125], all_err_p[120:125], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[125:130], all_err_n[125:130], all_err_p[125:130], alpha = 0.1, color = 'b')
    plt.fill_betweenx(question_numbers[130:135], all_err_n[130:135], all_err_p[130:135], alpha = 0.1, color = 'b')

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 15.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 20.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 25.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 30.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 35.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 40.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 45.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 50.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 55.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 60.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 65.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 70.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 75.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 80.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 85.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 90.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 95.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 100.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 105.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 110.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 115.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 120.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 125.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 130.5, color = 'm', ls = '--', lw = 0.3)

    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)
    
    plt.ylim(0.5, 135.5)
    plt.xlim(0, 1)
    plt.yticks(question_numbers, question_names)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False, labelright = True)
    plt.ylabel(r'$\leftarrow$ Noise', fontsize = 14)
    plt.xlabel(r'Score')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_large.png'

    #plt.legend(loc = 'upper right')
    plt.gca().invert_yaxis()
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------- Plot relative standard deviations -----------------------------------------------------------------

    fig = plt.figure('Parallel', figsize = (6, 8), dpi = 500)
    plt.subplots_adjust(left = 0.3)

    relative_err_lorentz = np.divide(all_std_lorentz, np.nan_to_num(all_scores_lorentz, copy = False, nan = 0))

    plt.plot(relative_err_lorentz[0:5], question_numbers[0:5], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[5:10], question_numbers[5:10], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[10:15], question_numbers[10:15], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[15:20], question_numbers[15:20], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[20:25], question_numbers[20:25], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[25:30], question_numbers[25:30], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[30:35], question_numbers[30:35], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[35:40], question_numbers[35:40], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[40:45], question_numbers[40:45], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[45:50], question_numbers[45:50], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[50:55], question_numbers[50:55], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[55:60], question_numbers[55:60], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[60:65], question_numbers[60:65], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[65:70], question_numbers[65:70], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[70:75], question_numbers[70:75], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[75:80], question_numbers[75:80], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[80:85], question_numbers[80:85], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[85:90], question_numbers[85:90], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[90:95], question_numbers[90:95], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[95:100], question_numbers[95:100], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[100:105], question_numbers[100:105], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[105:110], question_numbers[105:110], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[110:115], question_numbers[110:115], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[115:120], question_numbers[115:120], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[120:125], question_numbers[120:125], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[125:130], question_numbers[125:130], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)
    plt.plot(relative_err_lorentz[130:135], question_numbers[130:135], marker = 'o', linestyle = '-', color = 'k', linewidth = 0.5, markersize = 1)

    empty = np.zeros(135)
    plt.fill_betweenx(question_numbers, relative_err_lorentz, empty, alpha = 0.1, color = 'r')

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 15.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 20.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 25.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 30.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 35.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 40.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 45.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 50.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 55.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 60.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 65.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 70.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 75.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 80.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 85.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 90.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 95.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 100.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 105.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 110.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 115.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 120.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 125.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 130.5, color = 'm', ls = '--', lw = 0.3)
    
    plt.ylim(0.5, 135.5)
    plt.xlim(0.1, 10)
    plt.xscale("log")
    plt.yticks(question_numbers, question_names)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False)
    plt.xlabel(r'Normed standard deviation in score')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_error_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_error_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_error_large.png'

    #plt.legend(loc = 'upper right')
    plt.gca().invert_yaxis()
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------- Plot rejected responses -----------------------------------------------------------------

    fig = plt.figure('Parallel', figsize = (6, 8), dpi = 500)
    plt.subplots_adjust(left = 0.3)
    
    plt.plot(all_invalid[0:5], question_numbers[0:5], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[5:10], question_numbers[5:10], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[10:15], question_numbers[10:15], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[15:20], question_numbers[15:20], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[20:25], question_numbers[20:25], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[25:30], question_numbers[25:30], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[30:35], question_numbers[30:35], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[35:40], question_numbers[35:40], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[40:45], question_numbers[40:45], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[45:50], question_numbers[45:50], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[50:55], question_numbers[50:55], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[55:60], question_numbers[55:60], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[60:65], question_numbers[60:65], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[65:70], question_numbers[65:70], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[70:75], question_numbers[70:75], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[75:80], question_numbers[75:80], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[80:85], question_numbers[80:85], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[85:90], question_numbers[85:90], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[90:95], question_numbers[90:95], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[95:100], question_numbers[95:100], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[100:105], question_numbers[100:105], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[105:110], question_numbers[105:110], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[110:115], question_numbers[110:115], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[115:120], question_numbers[115:120], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[120:125], question_numbers[120:125], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[125:130], question_numbers[125:130], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_invalid[130:135], question_numbers[130:135], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 15.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 20.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 25.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 30.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 35.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 40.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 45.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 50.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 55.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 60.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 65.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 70.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 75.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 80.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 85.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 90.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 95.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 100.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 105.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 110.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 115.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 120.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 125.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 130.5, color = 'm', ls = '--', lw = 0.3)

    plt.ylim(1, 136)
    plt.xlim(0, 1)
    plt.yticks(question_numbers, question_names)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False)
    plt.xlabel(r'Frequency of rejected responses')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_rejected_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_rejected_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_rejected_large.png'

    plt.gca().invert_yaxis()
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------- Plot tiered scores -----------------------------------------------------------------

    T1_scores /= 11; T2_scores /= 9; T3_scores /= 7

    tiers = [' ', ' ', r'Tier I', ' ', ' ']
    tiers.extend([' ', ' ', r'Tier II', ' ', ' '])
    tiers.extend([' ', ' ', r'Tier III', ' ', ' '])

    fig = plt.figure('Parallel', figsize = (3, 4), dpi = 500)
    plt.subplots_adjust(left = 0.3)

    plt.plot(T1_scores, question_numbers[0:5], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(T2_scores, question_numbers[5:10], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(T3_scores, question_numbers[10:15], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    
    plt.ylim(0.5, 15.5)
    plt.xlim(0, 1)
    plt.yticks(question_numbers[0:15], tiers)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False)
    plt.ylabel(r'$\leftarrow$ Noise')
    plt.xlabel(r'Score')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_large.png'

    #plt.legend(loc = 'upper right')
    plt.gca().invert_yaxis()
    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------- Save tiered scores (.csv) -----------------------

    data_concat = np.vstack((T1_scores, T2_scores, T3_scores)).T
    header = ['Tier I', 'Tier II', 'Tier III']
    file_csv = f'Evaluation/{model_resolution}_tiers.csv'
    np.savetxt(file_csv, data_concat, delimiter = ',', fmt = "%s", header = ','.join(header), comments = '')

    # ----------------------- Save scores (.csv) -----------------------

    header = ['Score']
    file_csv = f'Evaluation/{model_resolution}_scores.csv'
    np.savetxt(file_csv, all_scores_lorentz, delimiter = ',', fmt = "%s", header = ','.join(header), comments = '')

    return all_scores_lorentz, all_std_lorentz, model_resolution, model_name





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------- Score all questions (code) -----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def score_code_A1(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_A1 = 75
    elif size == 2:
        resolution_A1 = 125
    else:
        resolution_A1 = 250

    noise_level = int((Q_num - 1)*20)
    spectrum_name = f"Spectra/A1_r{int(resolution_A1)}_n{noise_level}"
    
    if size == 1:
        solution_name = "Prompts_small/A1/A1_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/A1/A1_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/A1/A1_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    response = fit_FD(E_array, k_array, spectrum)
    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, E_conv)
    plot_histogram_E_single([response], E_array, solution, E_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_A1(1, 1)
#print(score_lorentz)



def score_code_B1(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 6)*10)
    spectrum_name = f"Spectra/B1_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B1/B1_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B1/B1_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B1/B1_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = disp

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_dispersion_E([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz, disp

#score_erf, score_gauss, score_lorentz, disp = score_code_B1(1, 6)
#print(score_lorentz)



def score_code_B1_vF(size, Q_num, disp):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01
    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    vF_linear = np.abs(linear_band_params[0])

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 6)*10)
    spectrum_name = f"Spectra/B1_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B1/B1_vF_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num + 5)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B1/B1_vF_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num + 5)}.png"
    else:
        solution_name = "Prompts_large/B1/B1_vF_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num + 5)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")
    
    vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
    response = find_vF(disp, E_array, 1, 24.94)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, vF_sigma)
    axis_label = r'Fermi velocity (eV$\cdot\AA$)'
    plot_histogram_general_single([response], solution, vF_sigma, 6, axis_label, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_B1_vF(1, 6, disp)
#print(score_lorentz)



def score_code_B2(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 16)*10)
    spectrum_name = f"Spectra/B2_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B2/B2_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B2/B2_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B2/B2_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = disp

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_dispersion_E([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz, disp

#score_erf, score_gauss, score_lorentz, disp = score_code_B2(1, 16)
#print(score_lorentz)



def score_code_B2_vF(size, Q_num, disp):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_quadratic = np.abs(quadratic_band_params[0])

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 16)*10)
    spectrum_name = f"Spectra/B2_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B2/B2_vF_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num + 5)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B2/B2_vF_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num + 5)}.png"
    else:
        solution_name = "Prompts_large/B2/B2_vF_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num + 5)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")
    
    vF_sigma = vF_quadratic*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
    response = find_vF(disp, E_array, 2, 24.94)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, vF_sigma)
    axis_label = r'Fermi velocity (eV$\cdot\AA$)'
    plot_histogram_general_single([response], solution, vF_sigma, 6, axis_label, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_B2_vF(1, 16, disp)
#print(score_lorentz)



def score_code_B3(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 26)*10)
    spectrum_name = f"Spectra/B3_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B3/B3_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B3/B3_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B3/B3_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    #disp = fit_Ln(E_array, k_array, spectrum, 0.05, 30)
    disp = fit_L1_restrained(E_array, k_array, spectrum, 0.05, 50)
    response = disp

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_dispersion_E([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz, disp

#score_erf, score_gauss, score_lorentz, disp = score_code_B3(1, 26)
#print(score_lorentz)



def score_code_B3_vF(size, Q_num, disp):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01
    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    vF_linear = np.abs(linear_band_params[0])

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 26)*10)
    spectrum_name = f"Spectra/B3_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B3/B3_vF_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num + 5)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B3/B3_vF_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num + 5)}.png"
    else:
        solution_name = "Prompts_large/B3/B3_vF_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num + 5)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")
    
    vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array) - np.min(E_array))**2))
    response = find_vF(disp, E_array, 1, 24.94)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, vF_sigma)
    axis_label = r'Fermi velocity (eV$\cdot\AA$)'
    plot_histogram_general_single([response], solution, vF_sigma, 6, axis_label, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_B3_vF(1, 26, disp)
#print(score_lorentz)



def score_code_B4(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 36)*10)
    spectrum_name = f"Spectra/B4_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B4/B4_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B4/B4_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B4/B4_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)[1:]
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1_EDC(E_array, k_array, spectrum, 0.05)
    response = disp

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_dispersion_k([response], solution, E_array, k_array, E_conv, file_name)

    return score_erf, score_gauss, score_lorentz, disp

#score_erf, score_gauss, score_lorentz, disp = score_code_B4(1, 36)
#print(score_lorentz)



def score_code_B4_bbE(size, Q_num, disp):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 36)*10)
    spectrum_name = f"Spectra/B4_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B4/B4_bbE_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num + 5)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B4/B4_bbE_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num + 5)}.png"
    else:
        solution_name = "Prompts_large/B4/B4_bbE_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num + 5)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")
    
    response = find_bbE(disp, k_array, E_array, 1.65, 2.35)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, E_conv)
    plot_histogram_E_single([response], E_array, solution, E_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_B4_bbE(1, 36, disp)
#print(score_lorentz)



def score_code_B5(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 46)*10)
    spectrum_name = f"Spectra/B5_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B5/B5_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B5/B5_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B5/B5_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    response = find_Dirac(E_array, k_array, spectrum, 24.95, 24.98, 0.05)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, E_conv)
    plot_histogram_E_single([response], E_array, solution, E_conv, file_name)

    return score_erf[0], score_gauss[0], score_lorentz[0]

#score_erf, score_gauss, score_lorentz = score_code_B5(1, 46)
#print(score_lorentz)



def score_code_B6(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 51)*10)
    spectrum_name = f"Spectra/B6_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/B6/B6_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/B6/B6_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/B6/B6_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    response = find_SC_gap(E_array, k_array, spectrum, -0.25, -0.15)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, E_conv)
    E_lims = np.array([0, np.max(max(E_array))])
    plot_histogram_E_single([response], E_lims, solution, E_conv, file_name)

    return score_erf[0], score_gauss[0], score_lorentz[0]

#score_erf, score_gauss, score_lorentz = score_code_B6(1, 51)
#print(score_lorentz)



def score_code_C1(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 56)*5)
    spectrum_name = f"Spectra/C1_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/C1/C1_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/C1/C1_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/C1/C1_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = 2*gamma_array   # Multiply by two to get FWHM

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_linewidth([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_C1(1, 56)
#print(score_lorentz)



def score_code_C2(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 61)*5)
    spectrum_name = f"Spectra/C2_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/C2/C2_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/C2/C2_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/C2/C2_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = 2*gamma_array   # Multiply by two to get FWHM

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_linewidth([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_C2(1, 61)
#print(score_lorentz)



def score_code_C3(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 66)*5)
    spectrum_name = f"Spectra/C3_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/C3/C3_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/C3/C3_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/C3/C3_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = 2*gamma_array   # Multiply by two to get FWHM

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_linewidth([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_C3(1, 67)
#print(score_lorentz)



def score_code_C4(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 71)*5)
    spectrum_name = f"Spectra/C4_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/C4/C4_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/C4/C4_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/C4/C4_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = 2*gamma_array   # Multiply by two to get FWHM

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_linewidth([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_C4(1, 71)
#print(score_lorentz)



def score_code_C5(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 76)*5)
    spectrum_name = f"Spectra/C5_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/C5/C5_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/C5/C5_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/C5/C5_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = 2*gamma_array   # Multiply by two to get FWHM

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv)
    plot_linewidth([response], solution, E_array, k_array, k_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_C5(1, 76)
#print(score_lorentz)



def score_code_D1(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    if Q_num in range(81, 86):
        L = "_l05"
    elif Q_num in range(86, 91):
        L = "_l075"
    elif Q_num in range(91, 96):
        L = "_l10"
    elif Q_num in range(96, 101):
        L = "_l20"
    else:
        L = "_l50"

    noise_level = int(((Q_num - 81)%5)*10)
    spectrum_name = f"Spectra/D1_r{int(resolution_BCD)}_n{noise_level}" + L

    if size == 1:
        solution_name = "Prompts_small/D1/D1_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/D1/D1_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/D1/D1_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = find_1_phonon(gamma_array, E_array, 24.9, 24.975)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, E_conv*3)
    plot_histogram_E_single([response], E_array, solution, E_conv, file_name)

    return score_erf[0], score_gauss[0], score_lorentz[0]

#score_erf, score_gauss, score_lorentz = score_code_D1(1, 81)
#print(score_lorentz)



def score_code_D2(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 106)*10)
    spectrum_name = f"Spectra/D2_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/D2/D2_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/D2/D2_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/D2/D2_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = find_2_phonons(gamma_array, E_array, 24.9, 24.94, 24.94, 24.975)

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv*3)
    plot_histogram_E_double([response], E_array, solution[0], solution[1], E_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_D2(1, 105)
#print(score_lorentz)



def score_code_D3(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_BCD = 100
    elif size == 2:
        resolution_BCD = 200
    else:
        resolution_BCD = 300

    noise_level = int((Q_num - 111)*10)
    spectrum_name = f"Spectra/D3_r{int(resolution_BCD)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/D3/D3_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/D3/D3_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/D3/D3_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")
    k_array = read_array(spectrum_name + "_k.txt")[1:]
    E_array = read_array(spectrum_name + "_E.txt")

    disp, gamma_array = fit_L1(E_array, k_array, spectrum, 0.05)
    response = find_3_phonons(gamma_array, E_array, 24.85, 24.88, 24.88, 24.935, 24.935, 24.975)

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, E_conv*3)
    plot_histogram_E_triple([response], E_array, solution[0], solution[1], solution[2], E_conv, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_D3(1, 111)
#print(score_lorentz)



def score_code_E1(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_E1 = 80
    elif size == 2:
        resolution_E1 = 150
    else:
        resolution_E1 = 250

    noise_level = int((Q_num - 116)*10)
    spectrum_name = f"Spectra/E1_r{int(resolution_E1)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/E1/E1_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/E1/E1_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/E1/E1_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_floats(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")[1:,:]
    kx_array = read_array(spectrum_name + "_kx.txt")[1:]
    ky_array = read_array(spectrum_name + "_ky.txt")[1:]

    response = find_doping_1(kx_array, ky_array, spectrum)

    score_erf, score_gauss, score_lorentz = score_code_floats(solution, response, doping_sigma)
    plot_doping_single([response], solution, doping_sigma, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_E1(1, 116)
#print(score_lorentz)



def score_code_E2(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_E234 = 110
    elif size == 2:
        resolution_E234 = 220
    else:
        resolution_E234 = 350

    noise_level = int((Q_num - 121)*10)
    spectrum_name = f"Spectra/E2_r{int(resolution_E234)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/E2/E2_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/E2/E2_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/E2/E2_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")[1:,:]
    kx_array = read_array(spectrum_name + "_kx.txt")[1:]
    ky_array = read_array(spectrum_name + "_ky.txt")[1:]

    response = find_dopings_2(kx_array, ky_array, spectrum)

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, doping_sigma)
    plot_doping_double([response], solution[0], solution[1], doping_sigma, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_E2(1, 121)
#print(score_lorentz)



def score_code_E3(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_E234 = 110
    elif size == 2:
        resolution_E234 = 220
    else:
        resolution_E234 = 350

    noise_level = int((Q_num - 126)*10)
    spectrum_name = f"Spectra/E3_r{int(resolution_E234)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/E3/E3_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/E3/E3_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/E3/E3_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")[1:,:]
    kx_array = read_array(spectrum_name + "_kx.txt")[1:]
    ky_array = read_array(spectrum_name + "_ky.txt")[1:]

    response = find_dopings_3(kx_array, ky_array, spectrum)

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, doping_sigma)
    plot_doping_triple([response], solution[0], solution[1], solution[2], doping_sigma, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_E3(1, 125)
#print(score_lorentz)



def score_code_E4(size, Q_num):

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    if size == 1:
        resolution_E234 = 110
    elif size == 2:
        resolution_E234 = 220
    else:
        resolution_E234 = 350

    noise_level = int((Q_num - 131)*10)
    spectrum_name = f"Spectra/E4_r{int(resolution_E234)}_n{noise_level}"

    if size == 1:
        solution_name = "Prompts_small/E4/E4_S.txt"
        file_name = "Spectra_answers/Small" + f"-Q{int(Q_num)}.png"
    elif size == 2:
        solution_name = "Prompts_med/E4/E4_S.txt"
        file_name = "Spectra_answers/Med" + f"-Q{int(Q_num)}.png"
    else:
        solution_name = "Prompts_large/E4/E4_S.txt"
        file_name = "Spectra_answers/Large" + f"-Q{int(Q_num)}.png"
    
    solution = get_solution_arrays(solution_name)
    spectrum = read_spectrum(spectrum_name + "_sp.txt")[1:,:]
    kx_array = read_array(spectrum_name + "_kx.txt")[1:]
    ky_array = read_array(spectrum_name + "_ky.txt")[1:]

    response = find_dopings_3(kx_array, ky_array, spectrum)

    score_erf, score_gauss, score_lorentz = score_code_arrays(solution, response, doping_sigma)
    plot_doping_triple([response], solution[0], solution[1], solution[2], doping_sigma, file_name)

    return score_erf, score_gauss, score_lorentz

#score_erf, score_gauss, score_lorentz = score_code_E4(1, 131)
#print(score_lorentz)



def score_code_responses(size):

    all_scores_erf = np.zeros(135); all_scores_gauss = np.zeros(135); all_scores_lorentz = np.zeros(135)
    all_scores_erf.fill(np.nan); all_scores_gauss.fill(np.nan); all_scores_lorentz.fill(np.nan)

    T1_scores = np.zeros(5); T2_scores = np.zeros(5); T3_scores = np.zeros(5)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    if size == 1:
        resolution_A1 = 75
        resolution_BCD = 100
        resolution_E1 = 80
        resolution_E234 = 110
    elif size == 2:
        resolution_A1 = 125
        resolution_BCD = 200
        resolution_E1 = 150
        resolution_E234 = 220
    else:
        resolution_A1 = 250
        resolution_BCD = 300
        resolution_E1 = 250
        resolution_E234 = 350

    for Q_num in range(1, 6):
        # Questions A1
        score_erf, score_gauss, score_lorentz = score_code_A1(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz
    print("A1 complete")
    
    for Q_num in range(6, 11):
        # Questions B1
        score_erf, score_gauss, score_lorentz, disp = score_code_B1(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
        
        # Questions B1 (vF)
        score_erf, score_gauss, score_lorentz = score_code_B1_vF(size, Q_num, disp)
        all_scores_erf[int(Q_num + 4)] = score_erf; all_scores_gauss[int(Q_num + 4)] = score_gauss; all_scores_lorentz[int(Q_num + 4)] = score_lorentz
        T3_scores[(Q_num + 4)%5] += score_lorentz
    print("B1 complete")
    
    for Q_num in range(16, 21):
        # Questions B2
        score_erf, score_gauss, score_lorentz, disp = score_code_B2(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
        
        # Questions B2 (vF)
        score_erf, score_gauss, score_lorentz = score_code_B2_vF(size, Q_num, disp)
        all_scores_erf[int(Q_num + 4)] = score_erf; all_scores_gauss[int(Q_num + 4)] = score_gauss; all_scores_lorentz[int(Q_num + 4)] = score_lorentz
        T3_scores[(Q_num + 4)%5] += score_lorentz
    print("B2 complete")
    
    for Q_num in range(26, 31):
        # Questions B3
        score_erf, score_gauss, score_lorentz, disp = score_code_B3(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
        
        # Questions B3 (vF)
        score_erf, score_gauss, score_lorentz = score_code_B3_vF(size, Q_num, disp)
        all_scores_erf[int(Q_num + 4)] = score_erf; all_scores_gauss[int(Q_num + 4)] = score_gauss; all_scores_lorentz[int(Q_num + 4)] = score_lorentz
        T3_scores[(Q_num + 4)%5] += score_lorentz
    print("B3 complete")
    
    for Q_num in range(36, 41):
        # Questions B4
        score_erf, score_gauss, score_lorentz, disp = score_code_B4(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
        
        # Questions B4 (bbE)
        score_erf, score_gauss, score_lorentz = score_code_B4_bbE(size, Q_num, disp)
        all_scores_erf[int(Q_num + 4)] = score_erf; all_scores_gauss[int(Q_num + 4)] = score_gauss; all_scores_lorentz[int(Q_num + 4)] = score_lorentz
        T1_scores[(Q_num + 4)%5] += score_lorentz
    print("B4 complete")
    
    for Q_num in range(46, 51):
        # Questions B5
        score_erf, score_gauss, score_lorentz = score_code_B5(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz
    print("B5 complete")
    
    for Q_num in range(51, 56):
        # Questions B6
        score_erf, score_gauss, score_lorentz = score_code_B6(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz
    print("B6 complete")
    
    for Q_num in range(56, 61):
        # Questions C1
        score_erf, score_gauss, score_lorentz = score_code_C1(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
    print("C1 complete")
    
    for Q_num in range(61, 66):
        # Questions C2
        score_erf, score_gauss, score_lorentz = score_code_C2(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
    print("C2 complete")
    
    for Q_num in range(66, 71):
        # Questions C3
        score_erf, score_gauss, score_lorentz = score_code_C3(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
    print("C3 complete")
    
    for Q_num in range(71, 76):
        # Questions C4
        score_erf, score_gauss, score_lorentz = score_code_C4(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
    print("C4 complete")
    
    for Q_num in range(76, 81):
        # Questions C5
        score_erf, score_gauss, score_lorentz = score_code_C5(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T2_scores[(Q_num - 1)%5] += score_lorentz
    print("C5 complete")
    
    for Q_num in range(81, 106):
        # Questions D1
        score_erf, score_gauss, score_lorentz = score_code_D1(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz
    print("D1 complete")
    
    for Q_num in range(106, 111):
        # Questions D2
        score_erf, score_gauss, score_lorentz = score_code_D2(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz
    print("D2 complete")
    
    for Q_num in range(111, 116):
        # Questions D3
        score_erf, score_gauss, score_lorentz = score_code_D3(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T1_scores[(Q_num - 1)%5] += score_lorentz    
    print("D3 complete")
    
    for Q_num in range(116, 121):
        # Questions E1
        score_erf, score_gauss, score_lorentz = score_code_E1(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T3_scores[(Q_num - 1)%5] += score_lorentz
    print("E1 complete")
    
    for Q_num in range(121, 126):
        # Questions E2
        score_erf, score_gauss, score_lorentz = score_code_E2(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T3_scores[(Q_num - 1)%5] += score_lorentz
    print("E2 complete")
    
    for Q_num in range(126, 131):
        # Questions E3
        score_erf, score_gauss, score_lorentz = score_code_E3(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T3_scores[(Q_num - 1)%5] += score_lorentz
    print("E3 complete")
    
    for Q_num in range(131, 136):
        # Questions E4
        score_erf, score_gauss, score_lorentz = score_code_E4(size, Q_num)
        all_scores_erf[int(Q_num - 1)] = score_erf; all_scores_gauss[int(Q_num - 1)] = score_gauss; all_scores_lorentz[int(Q_num - 1)] = score_lorentz
        T3_scores[(Q_num - 1)%5] += score_lorentz
    print("E4 complete")

    results = np.concatenate((all_scores_erf, all_scores_gauss, all_scores_lorentz), axis = 0)
    eval_name = "Evaluation/Code"

    if size == 1:
        write_to_text(str(results), eval_name, "_small")
        resolution = r" (low res)"
    elif size == 2:
        write_to_text(str(results), eval_name, "_med")
        resolution = r" (med res)"
    else:
        write_to_text(str(results), eval_name, "_large")
        resolution = r" (high res)"
    
    question_names = [' ', ' ', r'Fermi Level (A1)', ' ', ' ']
    question_names.extend([' ', ' ', r'Linear (B1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Linear $v_F$ (B1)', '', ''])
    question_names.extend([' ', ' ', r'Quadratic (B2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Quadratic $v_F$ (B2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superstructure (B3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superstructure $v_F$ (B3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Band bottom (B4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Band bottom energy (B4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Dirac cone energy (B5)', ' ', ' '])
    question_names.extend([' ', ' ', r'Superconducting gap size (B6)', ' ', ' '])
    question_names.extend([' ', ' ', r'Impurity scattering (C1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Marginal Fermi liquid, MFL (C2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Fermi liquid, FL (C3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon + MFL (C4)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon + FL (C5)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 0.5$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 0.75$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 1$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 2$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Phonon, $\lambda = 5$ (D1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Two phonons (D2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Three phonons (D3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Cuprate, single layer (E1)', ' ', ' '])
    question_names.extend([' ', ' ', r'Cuprate, bilayer (E2)', ' ', ' '])
    question_names.extend([' ', ' ', r'Sr$_2$RuO$_4$ (E3)', ' ', ' '])
    question_names.extend([' ', ' ', r'Nickelate, trilayer (E4)', ' ', ' '])
    
    model = "Code"
    model_name = "Code (human-written)"
    model_resolution = model + resolution
    question_numbers = np.arange(1, 136)

    # ----------------------------------------------------------------- Plot main scores -----------------------------------------------------------------

    fig = plt.figure('Parallel', figsize = (6, 8), dpi = 500)
    plt.subplots_adjust(left = 0.3)

    plt.plot(all_scores_lorentz[0:5], question_numbers[0:5], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[5:10], question_numbers[5:10], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[10:15], question_numbers[10:15], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[15:20], question_numbers[15:20], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[20:25], question_numbers[20:25], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[25:30], question_numbers[25:30], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[30:35], question_numbers[30:35], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[35:40], question_numbers[35:40], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[40:45], question_numbers[40:45], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[45:50], question_numbers[45:50], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[50:55], question_numbers[50:55], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[55:60], question_numbers[55:60], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[60:65], question_numbers[60:65], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[65:70], question_numbers[65:70], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[70:75], question_numbers[70:75], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[75:80], question_numbers[75:80], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[80:85], question_numbers[80:85], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[85:90], question_numbers[85:90], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[90:95], question_numbers[90:95], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[95:100], question_numbers[95:100], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[100:105], question_numbers[100:105], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[105:110], question_numbers[105:110], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[110:115], question_numbers[110:115], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[115:120], question_numbers[115:120], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[120:125], question_numbers[120:125], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[125:130], question_numbers[125:130], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(all_scores_lorentz[130:135], question_numbers[130:135], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 15.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 20.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 25.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 30.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 35.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 40.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 45.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 50.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 55.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 60.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 65.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 70.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 75.5, color = 'b', ls = '--', lw = 0.3)
    plt.axhline(y = 80.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 85.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 90.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 95.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 100.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 105.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 110.5, color = 'r', ls = '--', lw = 0.3)
    plt.axhline(y = 115.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 120.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 125.5, color = 'm', ls = '--', lw = 0.3)
    plt.axhline(y = 130.5, color = 'm', ls = '--', lw = 0.3)

    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)
    
    plt.ylim(0.5, 135.5)
    plt.xlim(0, 1)
    plt.yticks(question_numbers, question_names)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False, labelright = True)
    plt.ylabel(r'$\leftarrow$ Noise', fontsize = 14)
    plt.xlabel(r'Score')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_large.png'

    #plt.legend(loc = 'upper right')
    plt.gca().invert_yaxis()
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------- Plot tiered scores -----------------------------------------------------------------

    T1_scores /= 11; T2_scores /= 9; T3_scores /= 7

    tiers = [' ', ' ', r'Tier I', ' ', ' ']
    tiers.extend([' ', ' ', r'Tier II', ' ', ' '])
    tiers.extend([' ', ' ', r'Tier III', ' ', ' '])

    fig = plt.figure('Parallel', figsize = (3, 4), dpi = 500)
    plt.subplots_adjust(left = 0.3)

    plt.plot(T1_scores, question_numbers[0:5], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(T2_scores, question_numbers[5:10], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)
    plt.plot(T3_scores, question_numbers[10:15], marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.axhline(y = 5.5, color = 'g', ls = '--', lw = 0.3)
    plt.axhline(y = 10.5, color = 'g', ls = '--', lw = 0.3)
    
    plt.ylim(0.5, 15.5)
    plt.xlim(0, 1)
    plt.yticks(question_numbers[0:15], tiers)
    plt.yticks(fontsize = 7)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False)
    plt.ylabel(r'$\leftarrow$ Noise')
    plt.xlabel(r'Score')
    plt.title(model_name)

    if size == 1:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_small.png'
    elif size == 2:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_med.png'
    else:
        full_path = f'/workspaces/physics-benchmark/client/Evaluation/' + model + '_tiers_large.png'

    #plt.legend(loc = 'upper right')
    plt.gca().invert_yaxis()
    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------- Save tiered scores (.csv) -----------------------

    data_concat = np.vstack((T1_scores, T2_scores, T3_scores)).T
    header = ['Tier I', 'Tier II', 'Tier III']
    file_csv = f'Evaluation/{model_resolution}_tiers.csv'
    np.savetxt(file_csv, data_concat, delimiter = ',', fmt = "%s", header = ','.join(header), comments = '')

    # ----------------------- Save scores (.csv) -----------------------

    header = ['Score']
    file_csv = f'Evaluation/{model_resolution}_scores.csv'
    np.savetxt(file_csv, all_scores_lorentz, delimiter = ',', fmt = "%s", header = ','.join(header), comments = '')

    return all_scores_lorentz





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Score batches of questions -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def score_models():

    scores_dict = {}
    scores_noiseless_dict = {}

    # Low resolution
    model_list_1 = ["code", "deepseek-reasoner", "o1-mini", "o1-low", "o1-medium", "o3-mini-low", "o3-mini-medium", "o3-low", "o3-medium", "o4-mini-low", "o4-mini-medium", "claude-3-7-sonnet-20250219", "gemini-2_0-flash", "o1-high", "o3-high", "o3-mini-high", "o4-mini-high", "gemini-2_5-pro-preview-03-25", "o3-pro-2025-06-10", "kimi-k2", "glm-4_5", "qwen3-235b-thinking", "gpt-oss-20b", "gpt-oss-120b"]
    model_name_1 = ["Code (human-written)", "DeepSeek-R1 (*)", "o1-mini", "o1 (low)", "o1 (medium)", "o3-mini (low)", "o3-mini (medium)", "o3 (low)", "o3 (medium)", "o4-mini (low)", "o4-mini (medium)", "Claude 3.7 Sonnet", "Gemini 2.0 Flash", "o1 (high)", "o3 (high)", "o3-mini (high)", "o4-mini (high)", "Gemini 2.5 Pro Preview", "o3-pro", "Kimi K2 (*)", "GLM-4.5 (*)", "Qwen3 (*)", "gpt-oss-20b (*)", "gpt-oss-120b (*)"]
    
    for model_num in range(len(model_list_1)):
        model = model_list_1[model_num]
        model_name = model_name_1[model_num]
        if model == "code":
            all_scores_lorentz = score_code_responses(1)
            all_std_lorentz = np.zeros(135)
            all_scores_noiseless_lorentz = all_scores_lorentz[::5]
            all_std_noiseless_lorentz = all_std_lorentz[::5]

            ave_scores_lorentz = np.sum(np.nan_to_num(all_scores_lorentz, nan = 0))/135
            ave_scores_noiseless_lorentz = np.sum(np.nan_to_num(all_scores_noiseless_lorentz, nan = 0))/27
            ave_std_lorentz = np.sum(np.nan_to_num(all_std_lorentz, nan = 0))/135
            ave_std_noiseless_lorentz = np.sum(np.nan_to_num(all_std_noiseless_lorentz, nan = 0))/27

            scores_dict.update({model_name: np.array([ave_scores_lorentz, ave_std_lorentz])})
            scores_noiseless_dict.update({model_name: np.array([ave_scores_noiseless_lorentz, ave_std_noiseless_lorentz])})
        else:
            all_scores_lorentz, all_std_lorentz, model_resolution, model_name = score_model_responses(model, 1, model_name)
            all_scores_noiseless_lorentz = all_scores_lorentz[::5]
            all_std_noiseless_lorentz = all_std_lorentz[::5]

            ave_scores_lorentz = np.sum(np.nan_to_num(all_scores_lorentz, nan = 0))/135
            ave_scores_noiseless_lorentz = np.sum(np.nan_to_num(all_scores_noiseless_lorentz, nan = 0))/27
            ave_std_lorentz = np.sum(np.nan_to_num(all_std_lorentz, nan = 0))/135
            ave_std_noiseless_lorentz = np.sum(np.nan_to_num(all_std_noiseless_lorentz, nan = 0))/27

            scores_dict.update({model_name: np.array([ave_scores_lorentz, ave_std_lorentz])})
            scores_noiseless_dict.update({model_name: np.array([ave_scores_noiseless_lorentz, ave_std_noiseless_lorentz])})

    # High resolution
    # model_list_3 = ["gemini-2_0-flash", "gemini-2_5-pro-preview-03-25"]
    # for model in model_list_3:
    #     all_scores_lorentz, all_std_lorentz, model_resolution, model_name = score_model_responses(model, 3)
    #     all_scores_noiseless_lorentz = all_scores_lorentz[::5]
    #     all_std_noiseless_lorentz = all_std_lorentz[::5]

    #     ave_scores_lorentz = np.sum(np.nan_to_num(all_scores_lorentz, nan = 0))/135
    #     ave_scores_noiseless_lorentz = np.sum(np.nan_to_num(all_scores_noiseless_lorentz, nan = 0))/27
    #     ave_std_lorentz = np.sum(np.nan_to_num(all_std_lorentz, nan = 0))/135
    #     ave_std_noiseless_lorentz = np.sum(np.nan_to_num(all_std_noiseless_lorentz, nan = 0))/27

    #     scores_dict.update({model_name: np.array([ave_scores_lorentz, ave_std_lorentz])})
    #     scores_noiseless_dict.update({model_name: np.array([ave_scores_noiseless_lorentz, ave_std_noiseless_lorentz])})

    # ----------------------- Plot noisefull leaderboard -----------------------
    
    scores_dict_sorted = dict(sorted(scores_dict.items(), key = lambda x:x[1][0], reverse = False))
    model_names = list(scores_dict_sorted.keys())
    list_results = list(scores_dict_sorted.values())
    scores = [arr[0] for arr in list_results]
    stds = [arr[1] for arr in list_results]

    fig = plt.figure('Parallel', figsize = (8, 10), dpi = 500)
    plt.subplots_adjust(left = 0.3)
    plt.barh(model_names, scores, align = 'center')
    plt.title('Leaderboard')
    plt.yticks(model_names) 
    plt.xlabel('Score')
    plt.xlim(0, 1)
    plt.gca().set_xlim(0, 1)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)

    plt.savefig('Evaluation/All_leaderboard.png', bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------- Plot noiseless leaderboard -----------------------
    
    scores_noiseless_dict_sorted = dict(sorted(scores_noiseless_dict.items(), key = lambda x:x[1][0], reverse = False))
    noiseless_model_names = list(scores_noiseless_dict_sorted.keys())
    noiseless_list_results = list(scores_noiseless_dict_sorted.values())
    noiseless_scores = [arr[0] for arr in noiseless_list_results]
    noiseless_stds = [arr[1] for arr in noiseless_list_results]

    fig = plt.figure('Parallel', figsize = (8, 10), dpi = 500)
    plt.subplots_adjust(left = 0.3)
    plt.barh(noiseless_model_names, noiseless_scores, align = 'center')
    plt.title('Leaderboard (noiseless)')
    plt.yticks(noiseless_model_names) 
    plt.xlabel('Score')
    plt.xlim(0, 1)
    plt.gca().set_xlim(0, 1)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.6, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)

    plt.savefig('Evaluation/All_noiseless_leaderboard.png', bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------- Plot both leaderboards -----------------------

    scores_noiseless_dict_sorted_2 = {}
    for key in scores_dict_sorted:
        scores_noiseless_dict_sorted_2[key] = scores_noiseless_dict[key]
    noiseless_model_names_2 = list(scores_noiseless_dict_sorted_2.keys())
    noiseless_list_results_2 = list(scores_noiseless_dict_sorted_2.values())
    noiseless_scores_2 = [arr[0] for arr in noiseless_list_results_2]
    noiseless_stds_2 = [arr[1] for arr in noiseless_list_results_2]

    fig = plt.figure('Parallel', figsize = (8, 4.5), dpi = 500)
    plt.subplots_adjust(left = 0.3)
    plt.barh(noiseless_model_names_2, noiseless_scores_2, align = 'center', color = 'tab:blue', label = r'Noiseless')
    plt.barh(model_names, scores, align = 'center', color = 'tab:red', label = r'All spectra')
    plt.title('Leaderboard')
    plt.yticks(noiseless_model_names)
    plt.xlabel('Score')
    plt.xlim(0, 0.6)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.axvline(x = 0.05, color = 'k', ls = '--', lw = 0.1)
    plt.axvline(x = 0.1, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.15, color = 'k', ls = '--', lw = 0.1)
    plt.axvline(x = 0.2, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.25, color = 'k', ls = '--', lw = 0.1)
    plt.axvline(x = 0.3, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.35, color = 'k', ls = '--', lw = 0.1)
    plt.axvline(x = 0.4, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.45, color = 'k', ls = '--', lw = 0.1)
    plt.axvline(x = 0.5, color = 'k', ls = '--', lw = 0.3)
    plt.axvline(x = 0.55, color = 'k', ls = '--', lw = 0.1)
    #plt.axvline(x = 0.8, color = 'k', ls = '--', lw = 0.3)
    plt.legend(loc = 'lower right')

    plt.savefig('Evaluation/All_leaderboard_both.png', bbox_inches = 'tight')
    plt.show()
    plt.close(fig)

    # ----------------------- Save both leaderboards (.csv) -----------------------

    leaderboard_concat = np.vstack((noiseless_model_names_2, scores, noiseless_scores_2)).T
    header = ['Model', 'All scores', 'Noiseless scores']
    file_csv = f'Evaluation/Leaderboards.csv'
    np.savetxt(file_csv, leaderboard_concat, delimiter = ',', fmt = "%s", header = ','.join(header), comments = '')
    
    return

def plot_models():
    #plot_model_responses("deepseek-reasoner", 1)
    #plot_model_responses("o1-mini", 1)
    #plot_model_responses("o1-low", 1)
    #plot_model_responses("o1-medium", 1)
    #plot_model_responses("o1-high", 1)
    #plot_model_responses("o3-mini-low", 1)
    #plot_model_responses("o3-mini-medium", 1)
    #plot_model_responses("o3-mini-high", 1)
    #plot_model_responses("o3-low", 1)
    #plot_model_responses("o3-medium", 1)
    #plot_model_responses("o3-high", 1)
    #plot_model_responses("o4-mini-low", 1)
    #plot_model_responses("o4-mini-medium", 1)
    #plot_model_responses("o4-mini-high", 1)
    #plot_model_responses("claude-3-7-sonnet-20250219", 1)
    #plot_model_responses("gemini-2_5-pro-preview-03-25", 1)
    #plot_model_responses("gemini-2_0-flash", 1)
    #plot_model_responses("gemini-2_5-pro-preview-03-25", 3)
    #plot_model_responses("gemini-2_0-flash", 3)
    plot_model_responses("o3-pro-2025-06-10", 1)
    plot_model_responses("kimi-k2", 1)
    plot_model_responses("glm-4_5", 1)
    plot_model_responses("qwen3-235b-thinking", 1)
    plot_model_responses("gpt-oss-20b", 1)
    plot_model_responses("gpt-oss-120b", 1)
    
    return

score_models()

#plot_models()



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------- Score resolution dependence ----------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def score_A1_single_resolution_iter(model, resolution_array, noise_ratio):
    
    res_length = len(resolution_array)
    all_scores_lorentz = np.zeros(res_length); all_std_lorentz = np.zeros(res_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan)
    noise_int = round(100*noise_ratio)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    scores_dict = {}; scores_dict["Pixels"] = []; scores_dict["Score"] = []

    for res_num in range(res_length):
        resolution = int(resolution_array[res_num])
        response_prefix = "Responses_single/"
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/A1/A1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-A1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz[int(res_num)] = score_lorentz
        all_std_lorentz[int(res_num)] = std_lorentz
        scores_dict["Pixels"].append(resolution**2)
        scores_dict["Score"].append(score_lorentz_all)
    
    df = pd.DataFrame(scores_dict)

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_A1")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    resolution_sq = np.square(resolution_array)
    plt.plot(resolution_sq, all_scores_lorentz, marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.ylim(0.5*np.min(all_scores_lorentz), 1)
    #plt.ylim(0, 1)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel(r'Score')
    plt.xlabel(r'Pixels')
    plt.gca().ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText = True)
    plt.title(model + " (A1)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_A1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)

    fig_regular = create_violin_plot(scores_dict)
    fig_scaled = create_scaled_violin_plot(scores_dict)
    
    # Save the regular plot
    regular_path = f'/workspaces/physics-benchmark/client/Evaluation_single/{model}_A1_violin.png'
    plt.figure(fig_regular.number)
    plt.savefig(regular_path)
    
    # Save the scaled plot
    scaled_path = f'/workspaces/physics-benchmark/client/Evaluation_single/{model}_A1_scaled_violin.png'
    plt.figure(fig_scaled.number)
    plt.savefig(scaled_path)
    
    plt.close(fig_regular)
    plt.close(fig_scaled)
    return



def score_B1_single_resolution_iter(model, resolution_array, noise_ratio):
    
    res_length = len(resolution_array)
    all_scores_lorentz = np.zeros(res_length); all_std_lorentz = np.zeros(res_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan)
    noise_int = round(100*noise_ratio)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for res_num in range(res_length):
        resolution = int(resolution_array[res_num])
        response_prefix = "Responses_single/"
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B1/B1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B1_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays(solution_name, response_name, k_conv)
        all_scores_lorentz[int(res_num)] = score_lorentz
        all_std_lorentz[int(res_num)] = std_lorentz

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_B1")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    resolution_sq = np.square(resolution_array)
    plt.plot(resolution_sq, all_scores_lorentz, marker = 'o', linestyle = '-', color = 'b', linewidth = 0.5, markersize = 1)

    plt.ylim(0.5*np.min(all_scores_lorentz), 1)
    #plt.ylim(0, 1)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel(r'Score')
    plt.xlabel(r'Pixels')
    plt.gca().ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText = True)
    plt.title(model + " (B1)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_B1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def score_A1_B1_single_resolution_iter(model, resolution_array_A1, resolution_array_B1, noise_ratio):
    
    res_length = len(resolution_array_A1)
    all_scores_lorentz = np.zeros(res_length); all_std_lorentz = np.zeros(res_length); all_err_lorentz_n = np.zeros(res_length); all_err_lorentz_p = np.zeros(res_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan); all_err_lorentz_n.fill(np.nan); all_err_lorentz_p.fill(np.nan)
    noise_int = round(100*noise_ratio)
    all_scores_full = np.empty((0,)); all_res_full = np.empty((0,))

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for res_num in range(res_length):
        resolution = int(resolution_array_A1[res_num])
        response_prefix = "Responses_single/"
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/A1/A1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-A1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz[int(res_num)] = score_lorentz
        all_std_lorentz[int(res_num)] = std_lorentz
        all_err_lorentz_n[int(res_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p[int(res_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_scores_full = np.concatenate((all_scores_full, score_lorentz_all))
        all_res_full = np.concatenate((all_res_full, np.ones(len(score_lorentz_all))*(resolution**2)))
        if all_err_lorentz_n[int(res_num)] < 0:
            all_err_lorentz_n[int(res_num)] = 0.1*sorted(set(score_lorentz_all))[1]
    
    res_length_2 = len(resolution_array_B1)
    all_scores_lorentz_2 = np.zeros(res_length_2); all_std_lorentz_2 = np.zeros(res_length_2); all_err_lorentz_n_2 = np.zeros(res_length_2); all_err_lorentz_p_2 = np.zeros(res_length_2)
    all_scores_lorentz_2.fill(np.nan); all_std_lorentz_2.fill(np.nan); all_err_lorentz_n_2.fill(np.nan); all_err_lorentz_p_2.fill(np.nan)
    all_scores_full_2 = np.empty((0,)); all_res_full_2 = np.empty((0,))
    
    for res_num in range(res_length_2):
        resolution = int(resolution_array_B1[res_num])
        response_prefix = "Responses_single/"
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B1/B1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B1_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_lorentz_2[int(res_num)] = score_lorentz
        all_std_lorentz_2[int(res_num)] = std_lorentz
        all_err_lorentz_n_2[int(res_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p_2[int(res_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_scores_full_2 = np.concatenate((all_scores_full_2, score_lorentz_all))
        all_res_full_2 = np.concatenate((all_res_full_2, np.ones(len(score_lorentz_all))*(resolution**2)))
        if all_err_lorentz_n_2[int(res_num)] < 0:
            all_err_lorentz_n_2[int(res_num)] = 0.1*sorted(set(score_lorentz_all))[1]

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    #eval_name = "Evaluation_single/" + model
    #write_to_text(str(results), eval_name, "_A1")

    #fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    fig = plt.figure('Parallel', figsize = (6, 4.5), dpi = 500)
    resolution_sq_A1 = np.square(resolution_array_A1)
    plt.plot(resolution_sq_A1, all_scores_lorentz, linestyle = '-', color = 'b', linewidth = 0.75, label = 'A1')
    #plt.plot(resolution_sq_A1, all_err_lorentz_n, linestyle = ':', color = 'b', linewidth = 0.25)
    #plt.plot(resolution_sq_A1, all_err_lorentz_p, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.scatter(all_res_full, all_scores_full, color = 'b', s = 0.2, alpha = 0.25)
    plt.fill_between(resolution_sq_A1, all_err_lorentz_n, all_err_lorentz_p, alpha = 0.1, color = 'b')

    results = np.concatenate((all_scores_lorentz_2, all_std_lorentz_2), axis = 0)
    #eval_name = "Evaluation_single/" + model
    #write_to_text(str(results), eval_name, "_B1")

    resolution_sq_B1 = np.square(resolution_array_B1)
    plt.plot(resolution_sq_B1, all_scores_lorentz_2, linestyle = '-', color = 'r', linewidth = 0.75, label = 'B1')
    #plt.plot(resolution_sq_B1, all_err_lorentz_n_2, linestyle = ':', color = 'r', linewidth = 0.25)
    #plt.plot(resolution_sq_B1, all_err_lorentz_p_2, linestyle = ':', color = 'r', linewidth = 0.25)
    plt.scatter(all_res_full_2, all_scores_full_2, color = 'r', s = 0.2, alpha = 0.25)
    plt.fill_between(resolution_sq_B1, all_err_lorentz_n_2, all_err_lorentz_p_2, alpha = 0.1, color = 'r')

    y_floor = 10**(math.floor(np.log10(0.5*np.min(all_scores_lorentz_2))))
    plt.ylim(y_floor, 1)
    #plt.ylim(0, 1)
    plt.xlim(0, 1.02*np.max(resolution_sq_B1))
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel(r'Score')
    plt.xlabel(r'Pixels')
    plt.grid(axis = 'y', which = 'minor', linestyle = '--', linewidth = 0.1)
    plt.grid(axis = 'y', which = 'major', linestyle = '-', linewidth = 0.25)
    plt.gca().ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText = True)
    plt.title("Gemini 2.5 Pro Preview")
    plt.legend(loc = 'upper right')
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_A1_B1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



#resolution_array_B1 = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200, 225, 250, 275, 300])
#score_B1_single_resolution_iter("gemini-2_5-pro-preview-03-25", resolution_array_B1, 0)

#resolution_array_o3 = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200])
#score_B1_single_resolution_iter("o3-high", resolution_array_o3, 0)

#resolution_array_A1 = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200, 225, 250])
#score_A1_single_resolution_iter("gemini-2_5-pro-preview-03-25", resolution_array_A1, 0)

#score_A1_B1_single_resolution_iter("gemini-2_5-pro-preview-03-25", resolution_array_A1, resolution_array_B1, 0)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------- Score noise dependence --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def score_A1_single_noise_iter(model, resolution, noise_array):
    
    noise_length = len(noise_array)
    all_scores_lorentz = np.zeros(noise_length); all_std_lorentz = np.zeros(noise_length); all_err_lorentz_n = np.zeros(noise_length); all_err_lorentz_p = np.zeros(noise_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan); all_err_lorentz_n.fill(np.nan); all_err_lorentz_p.fill(np.nan)
    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/A1/A1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-A1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz[int(noise_num)] = score_lorentz
        all_std_lorentz[int(noise_num)] = std_lorentz
        all_err_lorentz_n[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n[int(noise_num)] < 0:
            all_err_lorentz_n[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_A1")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(noise_array, all_scores_lorentz, linestyle = '-', color = 'b', linewidth = 0.75)
    plt.plot(noise_array, all_err_lorentz_n, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.plot(noise_array, all_err_lorentz_p, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n, all_err_lorentz_p, alpha = 0.1, color = 'b')

    plt.ylim(0, 1)
    plt.xlim(0, np.max(noise_array))
    plt.ylabel(r'Score')
    plt.xlabel(r'Noise ratio')
    plt.grid(axis = 'y', linestyle = '--', linewidth = 0.25)
    plt.title("Gemini 2.5 Pro Preview (A1)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_Noise_A1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def score_B1_single_noise_iter(model, resolution, noise_array):
    
    noise_length = len(noise_array)
    all_scores_lorentz = np.zeros(noise_length); all_std_lorentz = np.zeros(noise_length); all_err_lorentz_n = np.zeros(noise_length); all_err_lorentz_p = np.zeros(noise_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan); all_err_lorentz_n.fill(np.nan); all_err_lorentz_p.fill(np.nan)
    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B1/B1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B1_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_lorentz[int(noise_num)] = score_lorentz
        all_std_lorentz[int(noise_num)] = std_lorentz
        all_err_lorentz_n[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n[int(noise_num)] < 0:
            all_err_lorentz_n[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_B1")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(noise_array, all_scores_lorentz, linestyle = '-', color = 'b', linewidth = 0.75)
    plt.plot(noise_array, all_err_lorentz_n, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.plot(noise_array, all_err_lorentz_p, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n, all_err_lorentz_p, alpha = 0.1, color = 'b')

    plt.ylim(0, 1.1*np.max(all_err_lorentz_p))
    plt.xlim(0, np.max(noise_array))
    plt.ylabel(r'Score')
    plt.xlabel(r'Noise ratio')
    plt.title("Gemini 2.5 Pro Preview (B1)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_Noise_B1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def score_B2_single_noise_iter(model, resolution, noise_array):
    
    noise_length = len(noise_array)
    all_scores_lorentz = np.zeros(noise_length); all_std_lorentz = np.zeros(noise_length); all_err_lorentz_n = np.zeros(noise_length); all_err_lorentz_p = np.zeros(noise_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan); all_err_lorentz_n.fill(np.nan); all_err_lorentz_p.fill(np.nan)
    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B2/B2_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B2_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_lorentz[int(noise_num)] = score_lorentz
        all_std_lorentz[int(noise_num)] = std_lorentz
        all_err_lorentz_n[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n[int(noise_num)] < 0:
            all_err_lorentz_n[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_B2")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(noise_array, all_scores_lorentz, linestyle = '-', color = 'b', linewidth = 0.75)
    plt.plot(noise_array, all_err_lorentz_n, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.plot(noise_array, all_err_lorentz_p, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n, all_err_lorentz_p, alpha = 0.1, color = 'b')

    plt.ylim(0, 1.1*np.max(all_err_lorentz_p))
    plt.xlim(0, np.max(noise_array))
    plt.ylabel(r'Score')
    plt.xlabel(r'Noise ratio')
    plt.title("Gemini 2.5 Pro Preview (B2)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_Noise_B2.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def score_D1_L1_single_noise_iter(model, resolution, noise_array):
    
    noise_length = len(noise_array)
    all_scores_lorentz = np.zeros(noise_length); all_std_lorentz = np.zeros(noise_length); all_err_lorentz_n = np.zeros(noise_length); all_err_lorentz_p = np.zeros(noise_length)
    all_scores_lorentz.fill(np.nan); all_std_lorentz.fill(np.nan); all_err_lorentz_n.fill(np.nan); all_err_lorentz_p.fill(np.nan)
    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/D1/D1_L1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz[int(noise_num)] = score_lorentz
        all_std_lorentz[int(noise_num)] = std_lorentz
        all_err_lorentz_n[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n[int(noise_num)] < 0:
            all_err_lorentz_n[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    results = np.concatenate((all_scores_lorentz, all_std_lorentz), axis = 0)
    eval_name = "Evaluation_single/" + model
    write_to_text(str(results), eval_name, "_D1_L1")

    fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    plt.plot(noise_array, all_scores_lorentz, linestyle = '-', color = 'b', linewidth = 0.75)
    plt.plot(noise_array, all_err_lorentz_n, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.plot(noise_array, all_err_lorentz_p, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n, all_err_lorentz_p, alpha = 0.1, color = 'b')

    plt.ylim(0, 1.1*np.max(all_err_lorentz_p))
    plt.xlim(0, np.max(noise_array))
    plt.ylabel(r'Score')
    plt.xlabel(r'Noise ratio')
    plt.title("Gemini 2.5 Pro Preview (D1)")
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_Noise_D1_L1.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



def score_four_single_noise_iter(model, resolution, noise_array):
    
    noise_length = len(noise_array)

    all_scores_lorentz_A1 = np.zeros(noise_length); all_std_lorentz_A1 = np.zeros(noise_length); all_err_lorentz_n_A1 = np.zeros(noise_length); all_err_lorentz_p_A1 = np.zeros(noise_length)
    all_scores_lorentz_A1.fill(np.nan); all_std_lorentz_A1.fill(np.nan); all_err_lorentz_n_A1.fill(np.nan); all_err_lorentz_p_A1.fill(np.nan)

    all_scores_lorentz_B1 = np.zeros(noise_length); all_std_lorentz_B1 = np.zeros(noise_length); all_err_lorentz_n_B1 = np.zeros(noise_length); all_err_lorentz_p_B1 = np.zeros(noise_length)
    all_scores_lorentz_B1.fill(np.nan); all_std_lorentz_B1.fill(np.nan); all_err_lorentz_n_B1.fill(np.nan); all_err_lorentz_p_B1.fill(np.nan)

    all_scores_lorentz_B2 = np.zeros(noise_length); all_std_lorentz_B2 = np.zeros(noise_length); all_err_lorentz_n_B2 = np.zeros(noise_length); all_err_lorentz_p_B2 = np.zeros(noise_length)
    all_scores_lorentz_B2.fill(np.nan); all_std_lorentz_B2.fill(np.nan); all_err_lorentz_n_B2.fill(np.nan); all_err_lorentz_p_B2.fill(np.nan)

    all_scores_lorentz_D1 = np.zeros(noise_length); all_std_lorentz_D1 = np.zeros(noise_length); all_err_lorentz_n_D1 = np.zeros(noise_length); all_err_lorentz_p_D1 = np.zeros(noise_length)
    all_scores_lorentz_D1.fill(np.nan); all_std_lorentz_D1.fill(np.nan); all_err_lorentz_n_D1.fill(np.nan); all_err_lorentz_p_D1.fill(np.nan)

    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/A1/A1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-A1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz_A1[int(noise_num)] = score_lorentz
        all_std_lorentz_A1[int(noise_num)] = std_lorentz
        all_err_lorentz_n_A1[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p_A1[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n_A1[int(noise_num)] < 0:
            all_err_lorentz_n_A1[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]
    
    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B1/B1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B1_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_lorentz_B1[int(noise_num)] = score_lorentz
        all_std_lorentz_B1[int(noise_num)] = std_lorentz
        all_err_lorentz_n_B1[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p_B1[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n_B1[int(noise_num)] < 0:
            all_err_lorentz_n_B1[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/B2/B2_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-B2_dispersion" + response_suffix + ".jsonl"
        solution_array, response_arrays = get_response_arrays(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz, invalid = score_response_arrays_all(solution_name, response_name, k_conv)
        all_scores_lorentz_B2[int(noise_num)] = score_lorentz
        all_std_lorentz_B2[int(noise_num)] = std_lorentz
        all_err_lorentz_n_B2[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p_B2[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n_B2[int(noise_num)] < 0:
            all_err_lorentz_n_B2[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"
        solution_name = f"Prompts_single/D1/D1_L1_r{resolution}_n{noise_int}" + "_S.txt"

        response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
        solution, responses = get_response_floats(solution_name, response_name)
        score_erf, score_gauss, score_lorentz, score_erf_all, score_gauss_all, score_lorentz_all, std_erf, std_gauss, std_lorentz = score_response_floats_all(solution_name, response_name, E_conv)
        all_scores_lorentz_D1[int(noise_num)] = score_lorentz
        all_std_lorentz_D1[int(noise_num)] = std_lorentz
        all_err_lorentz_n_D1[int(noise_num)] = score_lorentz - np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        all_err_lorentz_p_D1[int(noise_num)] = score_lorentz + np.absolute(std_lorentz/np.sqrt(len(score_lorentz_all)))
        if all_err_lorentz_n_D1[int(noise_num)] < 0:
            all_err_lorentz_n_D1[int(noise_num)] = 0.5*sorted(set(score_lorentz_all))[1]

    #fig = plt.figure('Parallel', figsize = (8, 6), dpi = 500)
    fig = plt.figure('Parallel', figsize = (6, 4.5), dpi = 500)

    plt.plot(noise_array, all_scores_lorentz_A1, linestyle = '-', color = 'b', linewidth = 0.75, label = 'A1')
    #plt.plot(noise_array, all_err_lorentz_n_A1, linestyle = ':', color = 'b', linewidth = 0.25)
    #plt.plot(noise_array, all_err_lorentz_p_A1, linestyle = ':', color = 'b', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n_A1, all_err_lorentz_p_A1, alpha = 0.05, color = 'b')

    plt.plot(noise_array, all_scores_lorentz_B1, linestyle = '-', color = 'g', linewidth = 0.75, label = 'B1')
    #plt.plot(noise_array, all_err_lorentz_n_B1, linestyle = ':', color = 'g', linewidth = 0.25)
    #plt.plot(noise_array, all_err_lorentz_p_B1, linestyle = ':', color = 'g', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n_B1, all_err_lorentz_p_B1, alpha = 0.05, color = 'g')

    plt.plot(noise_array, all_scores_lorentz_B2, linestyle = '-', color = 'r', linewidth = 0.75, label = 'B2')
    #plt.plot(noise_array, all_err_lorentz_n_B2, linestyle = ':', color = 'r', linewidth = 0.25)
    #plt.plot(noise_array, all_err_lorentz_p_B2, linestyle = ':', color = 'r', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n_B2, all_err_lorentz_p_B2, alpha = 0.05, color = 'r')

    plt.plot(noise_array, all_scores_lorentz_D1, linestyle = '-', color = 'c', linewidth = 0.75, label = 'D1')
    #plt.plot(noise_array, all_err_lorentz_n_D1, linestyle = ':', color = 'c', linewidth = 0.25)
    #plt.plot(noise_array, all_err_lorentz_p_D1, linestyle = ':', color = 'c', linewidth = 0.25)
    plt.fill_between(noise_array, all_err_lorentz_n_D1, all_err_lorentz_p_D1, alpha = 0.05, color = 'c')

    #y_floor = 10**(math.floor(np.log10(0.5*np.min(all_err_lorentz_n_D1))))
    y_floor = 0.005
    plt.ylim(y_floor, 1)
    plt.yscale('log')
    plt.xlim(0, np.max(noise_array))
    plt.ylabel(r'Score')
    plt.xlabel(r'Noise ratio')
    plt.grid(axis = 'y', which = 'minor', linestyle = '--', linewidth = 0.15)
    plt.grid(axis = 'y', which = 'major', linestyle = '-', linewidth = 0.25)
    plt.title("Gemini 2.5 Pro Preview")
    plt.legend(loc = 'upper right')
    full_path = f'/workspaces/physics-benchmark/client/Evaluation_single/' + model + '_Noise.png'
    plt.savefig(full_path)
    plt.show()
    plt.close(fig)
    return



#noise_array_total = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
#score_A1_single_noise_iter("gemini-2_5-pro-preview-03-25", 100, noise_array_total)
#score_B1_single_noise_iter("gemini-2_5-pro-preview-03-25", 100, noise_array_total)
#score_B2_single_noise_iter("gemini-2_5-pro-preview-03-25", 100, noise_array_total)
#score_D1_L1_single_noise_iter("gemini-2_5-pro-preview-03-25", 100, noise_array_total)

#score_four_single_noise_iter("gemini-2_5-pro-preview-03-25", 100, noise_array_total)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------- Plot noise dependence --------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def plot_noise_responses(model, resolution, noise_array, question_category):

    resolution = int(resolution)

    E_conv = 0.003; k_conv = 0.005; doping_sigma = 0.01
    E_factor = 1; k_factor = 1; doping_factor = 3
    E_conv *= E_factor; k_conv *= k_factor; doping_sigma *= doping_factor

    linear_band_params, k_int_range_linear = get_band_linear_params_d()
    quadratic_band_params, k_int_range_quadratic = get_band_quadratic_params_d()
    vF_linear = np.abs(linear_band_params[0]); vF_quadratic = np.abs(quadratic_band_params[0])

    noise_length = len(noise_array)

    for noise_num in range(noise_length):
        response_prefix = "Responses_single/"
        noise_int = round(100*noise_array[noise_num])
        response_suffix = f"_r{resolution}_n{noise_int}"

        if question_category == "A1":

            solution_name = f"Prompts_single/A1/A1_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-A1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-A1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array, k_array = get_spectrum_arrays_full(resolution)
            plot_histogram_E_single(responses, E_array, solution, E_conv, file_name)

        elif question_category == "B1":

            solution_name = f"Prompts_single/B1/B1_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B1_dispersion" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B1" + response_suffix + ".png"
        
            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)

        elif question_category == "B1_vF":

            solution_name = f"Prompts_single/B1/B1_vF_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B1_vF" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B1_vF" + response_suffix + ".png"

            E_array_cut, k_array = get_spectrum_arrays(resolution)
            vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array_cut) - np.min(E_array_cut))**2))
            solution, responses = get_response_floats(solution_name, response_name)
            axis_label = r'Fermi velocity (eV$\cdot\AA$)'
            plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
        
        elif question_category == "B2":

            solution_name = f"Prompts_single/B2/B2_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B2_dispersion" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B2" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "B2_vF":

            solution_name = f"Prompts_single/B2/B2_vF_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B2_vF" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B2_vF" + response_suffix + ".png"

            vF_sigma = vF_quadratic*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array_cut) - np.min(E_array_cut))**2))
            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
            
        elif question_category == "B3":

            solution_name = f"Prompts_single/B3/B3_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B3_dispersion" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B3" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_dispersion_E(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "B3_vF":

            solution_name = f"Prompts_single/B3/B3_vF_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B3_vF" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B3_vF" + response_suffix + ".png"

            vF_sigma = vF_linear*np.sqrt((k_conv/(np.max(k_array) - np.min(k_array)))**2 + (E_conv/(np.max(E_array_cut) - np.min(E_array_cut))**2))
            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_histogram_general_single(responses, solution, vF_sigma, 6, axis_label, file_name)
            
        elif question_category == "B4":

            solution_name = f"Prompts_single/B4/B4_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B4_dispersion" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B4" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_dispersion_k(response_arrays, solution_array, E_array_cut, k_array, E_conv, file_name)
            
        elif question_category == "B4_bbE":

            solution_name = f"Prompts_single/B4/B4_bbE_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B4_bbE" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B4_bbE" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "B5":

            solution_name = f"Prompts_single/B5/B5_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B5" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B5" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "B6":

            solution_name = f"Prompts_single/B6/B6_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-B6" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-B6" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_SC_arrays(resolution)
            E_lims = np.array([0, np.max(max(E_array_cut))])
            plot_histogram_E_single(responses, E_lims, solution, E_conv, file_name)
            
        elif question_category == "C1":

            solution_name = f"Prompts_single/C1/C1_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-C1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-C1" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "C2":

            solution_name = f"Prompts_single/C2/C2_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-C2" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-C2" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "C3":

            solution_name = f"Prompts_single/C3/C3_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-C3" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-C3" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "C4":

            solution_name = f"Prompts_single/C4/C4_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-C4" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-C4" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "C5":

            solution_name = f"Prompts_single/C5/C5_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-C5" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-C5" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays(resolution)
            plot_linewidth(response_arrays, solution_array, E_array_cut, k_array, k_conv, file_name)
            
        elif question_category == "D1_L05":

            solution_name = f"Prompts_single/D1/D1_L05_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "D1_L075":

            solution_name = f"Prompts_single/D1/D1_L075_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "D1_L1":

            solution_name = f"Prompts_single/D1/D1_L1_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "D1_L2":

            solution_name = f"Prompts_single/D1/D1_L2_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "D1_L5":

            solution_name = f"Prompts_single/D1/D1_L5_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_single(responses, E_array_cut, solution, E_conv, file_name)
            
        elif question_category == "D2":

            solution_name = f"Prompts_single/D2/D2_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D2" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D2" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_double(response_arrays, E_array_cut, solution_array[0], solution_array[1], E_conv, file_name)
            
        elif question_category == "D3":

            solution_name = f"Prompts_single/D3/D3_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-D3" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-D3" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            E_array_cut, k_array = get_spectrum_arrays_extended(resolution)
            plot_histogram_E_triple(response_arrays, E_array_cut, solution_array[0], solution_array[1], solution_array[2], E_conv, file_name)
            
        elif question_category == "E1":

            solution_name = f"Prompts_single/E1/E1_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-E1" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-E1" + response_suffix + ".png"

            solution, responses = get_response_floats(solution_name, response_name)
            plot_doping_single(responses, solution, doping_sigma, file_name)
            
        elif question_category == "E2":

            solution_name = f"Prompts_single/E2/E2_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-E2" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-E2" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            plot_doping_double(response_arrays, solution_array[0], solution_array[1], doping_sigma, file_name)
            
        elif question_category == "E3":

            solution_name = f"Prompts_single/E3/E3_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-E3" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-E3" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            plot_doping_triple(response_arrays, solution_array[0], solution_array[1], solution_array[2], doping_sigma, file_name)
            
        elif question_category == "E4":

            solution_name = f"Prompts_single/E4/E4_r{resolution}_n{noise_int}" + "_S.txt"
            response_name = response_prefix + model + "-E4" + response_suffix + ".jsonl"
            file_name = "Evaluation_single/" + model + "-E4" + response_suffix + ".png"

            solution_array, response_arrays = get_response_arrays(solution_name, response_name)
            plot_doping_triple(response_arrays, solution_array[0], solution_array[1], solution_array[2], doping_sigma, file_name)
            
    return



#noise_array_total = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
#plot_noise_responses("gemini-2_5-pro-preview-03-25", 100, noise_array_total, "A1")
#plot_noise_responses("gemini-2_5-pro-preview-03-25", 100, noise_array_total, "B1")
#plot_noise_responses("gemini-2_5-pro-preview-03-25", 100, noise_array_total, "B2")
#plot_noise_responses("gemini-2_5-pro-preview-03-25", 100, noise_array_total, "D1_L1")