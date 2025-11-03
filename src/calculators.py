import pandas as pd
import os
import uproot3 as uproot
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

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
        print("Warning: Division by zero or invalid value encountered in significance calculation. Returning 0.")
        return 0

def calculate_yields(data, channel):
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

def calculate_all_yields(data, signal, background, channels):

    results = {}
    
    for channel in channels:
        # Data yields
        data_yield, data_err = calculate_yields(data, channel)
        
        signal_yields = {}
        for name, df in signal.items():
            syield, serr = calculate_yields(df, channel)
            signal_yields[name] = (syield, serr)
        
        # Background yields
        background_yields = {}
        for name, df in background.items():
            byield, berr = calculate_yields(df, channel)
            background_yields[name] = (byield, berr)
        
        # Total signal and background
        total_signal = sum(v[0] for v in signal_yields.values())
        total_signal_err = math.sqrt(sum(v[1]**2 for v in signal_yields.values()))
        
        total_bkg = sum(v[0] for v in background_yields.values())
        total_bkg_err = math.sqrt(sum(v[1]**2 for v in background_yields.values()))
        
        # Significance calculation
        significance = calculate_significance(total_signal, total_bkg)
        
        # Store results
        ch_result = {
            'data': (data_yield, data_err),
            'signals': signal_yields,
            'backgrounds': background_yields,
            'total_signal': (total_signal, total_signal_err),
            'total_bkg': (total_bkg, total_bkg_err),
            'significance': significance
        }
        
        results[channel] = ch_result
    
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
        for key in channel_results['signals']:
            table_data.append([
                f"Signal - {key} ({channel})",
                f"{channel_results['signals'][key][0]:.2f}",
                f"{channel_results['signals'][key][1]:.2f}"
            ])
        
        # Add background rows
        for key in channel_results['backgrounds']:
            table_data.append([
                f"Background - {key} ({channel})",
                f"{channel_results['backgrounds'][key][0]:.2f}",
                f"{channel_results['backgrounds'][key][1]:.2f}"
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