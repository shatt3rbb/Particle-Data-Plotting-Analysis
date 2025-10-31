#!/usr/bin/python

import pandas as pd
import numpy as np
import uproot3 as uproot
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import math
import os
import sys
import yaml

allowed_event_types = ['all', 'ee', 'mm', 'mmm', 'mme', 'mee', 'eee']

def load_run_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['analysis']['region'], config['analysis']['control_type'], config['analysis']['event_type'], config['analysis']['scaling_option']

def initialize_unscaled_factors():
    #Return scaling factors set to 1.0 (no scaling)
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['scaling_factors']['Unscaled']

def get_scaling_factors(scaling_option):
    #Return scaling factors based on the scaling option
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    scaling_factors = config['scaling_factors']
    if scaling_option not in scaling_factors:
        raise SystemExit('The scaling option you have chosen is not correct. Available scaling options are: ' + str(list(scaling_factors.keys())))
    factors = scaling_factors[scaling_option]
    return factors

def get_SAMPLES_base_path():
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['BASE_PATH']

# Hardcoded Sample paths in config.yaml. Some .root fies might be missing.
def get_SAMPLE_PATHS():
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['SAMPLE_PATHS']

def get_VARIABLE_BINNING():
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['variable_binning']
    

def load_data_filtering_event_type(sample_path, event_type):
    """
    Load data from ROOT file and apply ONLY initial EVENT TYPE selection
    Returns:
        DataFrame with selected events
    """
    print(f'Loading data from: {sample_path}')
    file = uproot.open(sample_path)
    tree = file['tree']
    n_entries = tree.numentries
    print(f'File contains {n_entries} entries')

    # Mapping event_type to selection lambda
    selection_map = {
        'ee': lambda df: df[df['event_type'] == 1],
        'mm': lambda df: df[df['event_type'] == 0],
        'mmm': lambda df: df[df['event_3CR'] == 1],
        'mme': lambda df: df[df['event_3CR'] == 2],
        'mee': lambda df: df[df['event_3CR'] == 3],
        'eee': lambda df: df[df['event_3CR'] == 4],
        'all': lambda df: df  # No filtering
    }

    chunk_size = 400000
    chunks = []
    for i in range(0, n_entries, chunk_size):
        chunk = tree.pandas.df(entrystart=i, entrystop=min(i+chunk_size, n_entries), flatten=False)
        # Apply event type selection using the mapping
        chunk = selection_map[event_type](chunk)
        chunks.append(chunk)
        print(f'Processed {min(i+chunk_size, n_entries)} events')

    try:
        data = pd.concat(chunks)
    except ValueError:
        print('No events passed selection - returning empty DataFrame')
        data = tree.pandas.df(flatten=False).copy(deep=True)

    print(f'Finished loading {sample_path}')
    return data
    
def calculate_derived_variables(data):
    """
    Calculate additional physics variables needed for analysis
    
    Args:
        data: Input DataFrame with raw variables
    
    Returns:
        DataFrame with additional calculated variables
    """
    # Adjust phi values to be positive
    data.loc[(data["lepplus_phi"] < 0), "lepplus_phi"] += np.pi
    data.loc[(data["lepminus_phi"] < 0), "lepminus_phi"] += np.pi
    
    # Calculate derived variables
    data['dPhill'] = data['lepplus_phi'] - data['lepminus_phi']
    data['MetOZPt'] = data['met_tst'] / data['Z_pT']
    data['MetOHT_2'] = data['met_tst'] / (data['Z_pT'] + data['leading_jet_pt'] + data['second_jet_pt'])
    data['proshmo'] = (data['leading_jet_eta'] * data['second_jet_eta']) / np.abs((data['leading_jet_eta'] * data['second_jet_eta']))
    data['mT_ZZ'] = np.sqrt(2 * np.abs(data['Z_pT']) * np.abs(data['met_tst']) * (1 - np.cos(data['dMetZPhi'])))
    
    return data
    
def apply_analysis_cuts(data, control_type):
    """
    Apply analysis-specific cuts based on control region type from config.yaml
    """
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cuts = config['cuts'].get(control_type, {})
    data = data.reset_index(drop=True)  # Ensure index is sequential and starting from 0
    mask = pd.Series([True] * len(data))
    for var, expr in cuts.items():
        mask &= eval(expr, {}, {'x': data[var]})
    mask = mask.reset_index(drop=True)  # Align mask index with data
    return data[mask]
    
def apply_scaling_factors(samples, scaling_factors):
    """Apply scaling factors to the samples"""
    # Scale Z+jets samples
    samples['Z_jets_ee']['global_weight'] = np.where(
        samples['Z_jets_ee']['n_jets'] == 0,
        samples['Z_jets_ee']['global_weight'] * scaling_factors['sf_0_ee'],
        samples['Z_jets_ee']['global_weight']
    )
    samples['Z_jets_ee']['global_weight'] = np.where(
        samples['Z_jets_ee']['n_jets'] == 1,
        samples['Z_jets_ee']['global_weight'] * scaling_factors['sf_1_ee'],
        samples['Z_jets_ee']['global_weight']
    )
    samples['Z_jets_ee']['global_weight'] = np.where(
        samples['Z_jets_ee']['n_jets'] == 2,
        samples['Z_jets_ee']['global_weight'] * scaling_factors['sf_2_ee'],
        samples['Z_jets_ee']['global_weight']
    )
    samples['Z_jets_ee']['global_weight'] = np.where(
        samples['Z_jets_ee']['n_jets'] > 2,
        samples['Z_jets_ee']['global_weight'] * scaling_factors['sf_3_ee'],
        samples['Z_jets_ee']['global_weight']
    )
    
    # Similar scaling for muon channel
    samples['Z_jets_mumu']['global_weight'] = np.where(
        samples['Z_jets_mumu']['n_jets'] == 0,
        samples['Z_jets_mumu']['global_weight'] * scaling_factors['sf_0_mm'],
        samples['Z_jets_mumu']['global_weight']
    )
    samples['Z_jets_mumu']['global_weight'] = np.where(
        samples['Z_jets_mumu']['n_jets'] == 1,
        samples['Z_jets_mumu']['global_weight'] * scaling_factors['sf_1_mm'],
        samples['Z_jets_mumu']['global_weight']
    )
    samples['Z_jets_mumu']['global_weight'] = np.where(
        samples['Z_jets_mumu']['n_jets'] == 2,
        samples['Z_jets_mumu']['global_weight'] * scaling_factors['sf_2_mm'],
        samples['Z_jets_mumu']['global_weight']
    )
    samples['Z_jets_mumu']['global_weight'] = np.where(
        samples['Z_jets_mumu']['n_jets'] > 2,
        samples['Z_jets_mumu']['global_weight'] * scaling_factors['sf_3_mm'],
        samples['Z_jets_mumu']['global_weight']
    )
    
    # Scale other backgrounds
    samples['WZ']['global_weight'] *= scaling_factors['sf_3lCR']
    samples['top']['global_weight'] *= scaling_factors['sf_top']
    samples['ttbarV_ttbarVV']['global_weight'] *= scaling_factors['sf_top']
    samples['Wt']['global_weight'] *= scaling_factors['sf_top']
    samples['WW']['global_weight'] *= scaling_factors['sf_WW']
    
    return samples

def calculate_yields(data, channel):
    """
    Calculate event yields and statistical uncertainties
    
    Args:
        data: Input DataFrame
        channel: Event channel ('all', 'ee', 'mm', etc.)
    
    Returns:
        Tuple of (yield, uncertainty)
    """
    # Apply channel selection
    if channel == 'ee':
        data = data[data['event_type'] == 1]
    elif channel == 'mm':
        data = data[data['event_type'] == 0]
    elif channel == 'mmm':
        data = data[data['event_3CR'] == 1]
    elif channel == 'mme':
        data = data[data['event_3CR'] == 2]
    elif channel == 'mee':
        data = data[data['event_3CR'] == 3]
    elif channel == 'eee':
        data = data[data['event_3CR'] == 4]
    
    signal = data['global_weight'].sum()
    uncertainty = np.sqrt(np.square(data['global_weight']).sum())
    
    return signal, uncertainty
    
def calculate_significance(signal, background):
    """
    Calculate Poisson significance using log-likelihood ratio
    
    Args:
        signal: Signal yield
        background: Background yield
    
    Returns:
        Significance value
    """
    try:
        return np.sqrt(2 * ((signal + background) * np.log(1 + (signal / background)) - signal))
    except (ZeroDivisionError, ValueError):
        return 0


def calculate_all_yields(data, b_zjets, s_ewk, s_qcd, b_wz, b_top, b_ww, b_other, channels):
    """
    Calculate yields and significances for all processes and channels
    
    Returns dictionary with yields, uncertainties, and significances
    """
    results = {}
    
    for channel in channels:
        # Data yields
        data_yield, data_err = calculate_yields(data, channel)
        
        # Signal yields
        s_ewk_yield, s_ewk_err = calculate_yields(s_ewk, channel)
        s_qcd_yield, s_qcd_err = calculate_yields(s_qcd, channel)
        
        # Background yields
        b_wz_yield, b_wz_err = calculate_yields(b_wz, channel)
        b_zjets_yield, b_zjets_err = calculate_yields(b_zjets, channel)
        b_top_yield, b_top_err = calculate_yields(b_top, channel)
        b_ww_yield, b_ww_err = calculate_yields(b_ww, channel)
        b_other_yield, b_other_err = calculate_yields(b_other, channel)
        
        # Total signal and background
        total_signal = s_ewk_yield + s_qcd_yield
        total_signal_err = np.sqrt(s_ewk_err**2 + s_qcd_err**2)
        
        total_bkg = b_wz_yield + b_zjets_yield + b_top_yield + b_ww_yield + b_other_yield
        total_bkg_err = np.sqrt(b_wz_err**2 + b_zjets_err**2 + b_top_err**2 + b_ww_err**2 + b_other_err**2)
        
        # Significance calculation
        significance = calculate_significance(total_signal, total_bkg)
        
        # Store results
        results[channel] = {
            'data': (data_yield, data_err),
            'signal_ewk': (s_ewk_yield, s_ewk_err),
            'signal_qcd': (s_qcd_yield, s_qcd_err),
            'bkg_wz': (b_wz_yield, b_wz_err),
            'bkg_zjets': (b_zjets_yield, b_zjets_err),
            'bkg_top': (b_top_yield, b_top_err),
            'bkg_ww': (b_ww_yield, b_ww_err),
            'bkg_other': (b_other_yield, b_other_err),
            'total_signal': (total_signal, total_signal_err),
            'total_bkg': (total_bkg, total_bkg_err),
            'significance': significance
        }
    
    return results
    
def plot_distributions(data_list, names, variable, bins, region):
    """
    Create stacked distribution plots with data/MC comparison with numerical handling
    
    Args:
        data_list: List of DataFrames for each process
        names: List of process names
        variable: Physics variable to plot
        bins: Bin edges for histogram
        region: Analysis region
    
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    
    bins = np.array(bins)

    # Plot stacked MC
    weights = [d['global_weight'] for d in data_list[1:]]  # Skip data
    ax1.hist([d[variable] for d in data_list[1:]], bins=bins, weights=weights, stacked=True, label=names[1:])
    
    # Plot data
    data_hist, _ = np.histogram(data_list[0][variable], bins=bins, weights=data_list[0]['global_weight'])
    data_err = np.sqrt(np.histogram(data_list[0][variable], bins=bins, weights=data_list[0]['global_weight']**2)[0])
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax1.errorbar(bin_centers, data_hist, yerr=data_err, fmt='o', color='black', label='Data')
    
    # Add legend and labels
    ax1.legend()
    ax1.set_ylabel('Events')
    ax1.set_title(f'{variable} - {region}')
    ax1.grid()
    
    # Calculate MC histogram an errors
    mc_hist = np.sum([np.histogram(d[variable], bins=bins, weights=d['global_weight'])[0] for d in data_list[1:]], axis=0)
    mc_err = np.sqrt(np.sum([np.histogram(d[variable], bins=bins, weights=d['global_weight']**2)[0] for d in data_list[1:]], axis=0))

    # Add small epsilon to avoid true zeros in MC
    epsilon = 1e-10
    mc_hist = mc_hist + epsilon
    mc_err = mc_err + epsilon

    # Identify problematic bins for debugging
    problematic_bins = np.where((mc_hist <= epsilon) | (data_hist == 0))[0]
    #if len(problematic_bins) > 0:
        #print(f"Note: {len(problematic_bins)}/{len(bin_centers)} bins with near-zero MC or Data events in {variable}")
        #print(f"Bin centers with issues: {bin_centers[problematic_bins]}")

    # Safe division handling with proper error propagation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate ratio
        ratio = np.divide(data_hist, mc_hist)
        # Calculate relative errors
        rel_data_err = np.divide(data_err, data_hist, where=(data_hist > 0))
        rel_mc_err = np.divide(mc_err, mc_hist, where=(mc_hist > 0))
        # Calculate ratio error
        ratio_err = np.multiply(ratio, np.sqrt(rel_data_err**2 + rel_mc_err**2))
        # Replace problematic values
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
        ratio_err = np.nan_to_num(ratio_err, nan=0.0, posinf=0.0, neginf=0.0)

    # Set automatic y-limits with some padding
    valid_ratios = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(valid_ratios) > 0:
        y_min = max(0.5, 0.8 * np.min(valid_ratios))
        y_max = min(2.0, 1.2 * np.max(valid_ratios))
        ax2.set_ylim(y_min, y_max)
    else:
        ax2.set_ylim(0.5, 1.5)

    # Plot Ratio
    ax2.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='o', color='black')
    ax2.axhline(1, color='gray', linestyle='--')
    ax2.set_ylabel('Data/MC')
    ax2.set_xlabel(variable)
    ax2.grid()

    # Add marker for bins with zero MC events

    zero_mc_bins = np.where(mc_hist <= epsilon)[0]

    if len(zero_mc_bins) > 0:
        ax2.plot(bin_centers[zero_mc_bins], np.ones_like(zero_mc_bins), 'rx', markersize=10, label='Zero MC bins')
        ax2.legend()
    
    return fig
    
def create_yield_table(results, channels):
    """Create formatted yield table with uncertainties"""
    
    # Prepare table data
    table_data = []
    columns = ['Process', 'Yield', 'Uncertainty']
    
    for channel in channels:
        channel_results = results[channel]
        
        # Add data row
        table_data.append([
            f"Data ({channel})",
            f"{channel_results['data'][0]:.2f}",
            f"{channel_results['data'][1]:.2f}"
        ])
        
        # Add signal rows
        table_data.append([
            f"Signal EWK ({channel})",
            f"{channel_results['signal_ewk'][0]:.2f}",
            f"{channel_results['signal_ewk'][1]:.2f}"
        ])
        table_data.append([
            f"Signal QCD ({channel})",
            f"{channel_results['signal_qcd'][0]:.2f}",
            f"{channel_results['signal_qcd'][1]:.2f}"
        ])
        
        # Add background rows
        table_data.append([
            f"WZ ({channel})",
            f"{channel_results['bkg_wz'][0]:.2f}",
            f"{channel_results['bkg_wz'][1]:.2f}"
        ])
        table_data.append([
            f"Z+jets ({channel})",
            f"{channel_results['bkg_zjets'][0]:.2f}",
            f"{channel_results['bkg_zjets'][1]:.2f}"
        ])
        table_data.append([
            f"Top ({channel})",
            f"{channel_results['bkg_top'][0]:.2f}",
            f"{channel_results['bkg_top'][1]:.2f}"
        ])
        table_data.append([
            f"WW ({channel})",
            f"{channel_results['bkg_ww'][0]:.2f}",
            f"{channel_results['bkg_ww'][1]:.2f}"
        ])
        table_data.append([
            f"Other ({channel})",
            f"{channel_results['bkg_other'][0]:.2f}",
            f"{channel_results['bkg_other'][1]:.2f}"
        ])
        
        # Add totals
        table_data.append([
            f"Total Signal ({channel})",
            f"{channel_results['total_signal'][0]:.2f}",
            f"{channel_results['total_signal'][1]:.2f}"
        ])
        table_data.append([
            f"Total Background ({channel})",
            f"{channel_results['total_bkg'][0]:.2f}",
            f"{channel_results['total_bkg'][1]:.2f}"
        ])
        
        # Add significance
        table_data.append([
            f"Significance ({channel})",
            f"{channel_results['significance']:.2f}",
            ""
        ])
    
    # Create DataFrame
    yield_table = pd.DataFrame(table_data, columns=columns)
    return yield_table
    
def save_results(results, figures, channels, region, control_type, event_type, scaling_option, variable_binning):
    """Save all results to files"""
    output_dir = f"results/{region}_{control_type}_{event_type}_{scaling_option}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save yield tables
    yield_table = create_yield_table(results, channels)
    yield_table.to_csv(f"{output_dir}/yields.csv")
    
    # Save plots
    for i, fig in enumerate(figures):
        var_name = list(variable_binning.keys())[i]
        fig.savefig(f"{output_dir}/{var_name}.png")
        plt.close(fig)
    
    # Save additional information
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"Analysis Summary\n")
        f.write(f"Region: {region}\n")
        f.write(f"Control Type: {control_type}\n")
        f.write(f"Event Type: {event_type}\n")
        f.write(f"Scaling Option: {scaling_option}\n")
        for channel in channels:
            f.write(f"\nSignificance for {channel}: {results[channel]['significance']:.2f}\n")

if __name__ == "__main__":

    # Parse command line arguments or run initialization flags
    region, control_type, event_type, scaling_option = load_run_config('./src/config/config.yaml')

    # Get scaling factors
    if(scaling_option == 'Unscaled'):
        scaling_factors = initialize_unscaled_factors()
    else:
        scaling_factors = get_scaling_factors(scaling_option)

    # Load and process data
    samples = {}
    base_path = get_SAMPLES_base_path()

    #Simply checks the folder names under the SAMPLES folder in the base path. Does not actually check if these are valid folders that contain .root files.
    if(region not in os.listdir(base_path)):
        raise SystemExit('The region you have chosen is not correct. Available regions for the existing folder are: ' + str(os.listdir(base_path))[1:-1])
    else:
        base_path += region + '/'

    #Do a check if event_type is correctly input
    if(event_type not in allowed_event_types):
        raise SystemExit('The event type you have chosen is not correct. Available event type options are: ' + str(allowed_event_types))

    #Load data filtering by event type, calculate derived variables per event and apply analysis cuts depending on control type
    SAMPLE_PATHS = get_SAMPLE_PATHS()
    for sample_name, file_name in SAMPLE_PATHS.items():
        samples[sample_name] = load_data_filtering_event_type(base_path + file_name, event_type)
        samples[sample_name] = calculate_derived_variables(samples[sample_name])
        samples[sample_name] = apply_analysis_cuts(samples[sample_name], control_type)

    samples = apply_scaling_factors(samples, scaling_factors)

    #print(samples['WZ'].filter(['event_type', 'event_3CR']))

    signal_ewk = pd.concat([samples['llvvjj']])
    signal_qcd = pd.concat([samples['llvv']])

    background_wz = pd.concat([samples['WZ']])
    background_zjets = pd.concat([samples['Z_jets_mumu'], samples['Z_jets_ee']])
    background_top = pd.concat([samples['top'], samples['ttbarV_ttbarVV'], samples['Wt']])
    background_ww = pd.concat([samples['WW']])
    background_other = pd.concat([samples['llll'], samples['llqq'], samples['VVV'], samples['W_jets'], samples['Ztt']])

    #Preparing the channels list that will be used for yields
    if event_type == 'all':
        if '3lCR' in region:
            channels = ['all', 'mmm', 'mme', 'mee', 'eee']
        else:
            channels = ['all', 'ee', 'mm']
    else:
        channels = [event_type]

    # Calculate yields and significances
    yield_results = calculate_all_yields(samples['DATA'], background_zjets, signal_ewk, signal_qcd, background_wz, background_top, background_ww, background_other, channels)

    # Create plots for key variables
    plot_data = [samples['DATA'], signal_ewk, signal_qcd, background_zjets, background_wz, background_top, background_ww, background_other]
    plot_names = ['Data', 'EWK Signal', 'QCD Signal', 'Z+jets','WZ', 'Top', 'WW', 'Other']

    variable_binning = get_VARIABLE_BINNING()    

    figures = []
    for var, bins in variable_binning.items():
        fig = plot_distributions(plot_data, plot_names, var, bins, region)
        figures.append(fig)

    # Save results
    save_results(yield_results, figures, channels, region, control_type, event_type, scaling_option, variable_binning)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Region: {region}, Control Type: {control_type}, Event Type: {event_type}")
    for channel in channels:
        print(f"Significance for {channel}: {yield_results[channel]['significance']:.2f}")
    
    
    
        
    
