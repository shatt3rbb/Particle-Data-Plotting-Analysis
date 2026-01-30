import numpy as np
from matplotlib import pyplot as plt

def plot_signal_over_background(data_list, names, variable, bins, region):
    """
    Create a plot of N_events(Signal) / N_events(Background) per bin.
    
    Args:
        data_list: List of DataFrames for each process (index 0 is Data)
        names: List of process names
        variable: Physics variable to plot
        bins: Bin edges for histogram
        region: Analysis region
    
    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate Total Signal and Total Background histograms
    
    sig_hist_total = np.zeros(len(bin_centers))
    bk_hist_total = np.zeros(len(bin_centers))
    
    # helper to check if name implies signal or background, ignoring index 0 (Data)
    for i, (data, name) in enumerate(zip(data_list[1:], names[1:])):
        # Calculate histogram for this process
        weights = data['global_weight']
        hist, _ = np.histogram(data[variable], bins=bins, weights=weights)
        
        if "Signal" in name:
            sig_hist_total += hist
        elif "Background" in name:
            bk_hist_total += hist
        else:
            # Fallback: assume Background
            print(f"Warning: Unknown process type for {name} in S/B plot, treating as Background")
            bk_hist_total += hist

    # Calculate Ratio S/B
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sb_ratio = np.divide(sig_hist_total, bk_hist_total)
        # Replace inf/nan with 0 or appropriate value
        sb_ratio = np.nan_to_num(sb_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Plot Ratio
    ax1.plot(bin_centers, sb_ratio, marker='o', linestyle='-', linewidth=2, color='blue', label='S/B Ratio')
    
    ax1.set_ylabel('Signal / Background')
    ax1.set_xlabel(variable)
    ax1.set_title(f'Signal to Background Ratio - {variable} - {region}')
    ax1.grid(True)
    ax1.legend()
    
    return fig

def plot_separation_density(data_list, names, variable, bins, region):
    """
    Calculate and plot the separation density S_i = 1/2 * (s_i - b_i)^2 / (s_i + b_i) for each bin.
    s_i and b_i are the normalized bin contents (unity area normalization).
    
    Args:
        data_list: List of DataFrames for each process (index 0 is Data)
        names: List of process names
        variable: Physics variable to plot
        bins: Bin edges for histogram
        region: Analysis region
    
    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate Total Signal and Total Background histograms (Normalized)
    
    sig_hist_total = np.zeros(len(bin_centers))
    bk_hist_total = np.zeros(len(bin_centers))
    
    # First, collect counts to normalize later
    for i, (data, name) in enumerate(zip(data_list[1:], names[1:])):
        # Calculate histogram for this process
        weights = data['global_weight']
        hist, _ = np.histogram(data[variable], bins=bins, weights=weights)
        
        if "Signal" in name:
            sig_hist_total += hist
        elif "Background" in name:
            bk_hist_total += hist
        else:
            # Fallback
            print(f"Warning: Unknown process type for {name} in Separation plot, treating as Background")
            bk_hist_total += hist

    # Normalize s_i and b_i to unit area
    # Note: histograms are bin counts. To be PDFs, divide by (Sum * BinWidth).
    # However, separation formula usually works with "probabilities" p_i = count_i / total_count (sum of p_i = 1).
    # If bins have variable widths, we should be careful.
    # Standard formula S = 1/2 sum (p_s - p_b)^2/(p_s + p_b). This is sum over bins.
    # So p_s_i should be the fraction of total signal events in bin i.
    
    total_sig = np.sum(sig_hist_total)
    total_bk = np.sum(bk_hist_total)
    
    if total_sig > 0:
        s_i = sig_hist_total / total_sig
    else:
        s_i = sig_hist_total # zeros
        
    if total_bk > 0:
        b_i = bk_hist_total / total_bk
    else:
        b_i = bk_hist_total # zeros

    # Calculate Separation Density per bin
    # S_i = 0.5 * (s_i - b_i)^2 / (s_i + b_i)
    
    sep_density = np.zeros_like(s_i)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (s_i - b_i)**2
        denominator = s_i + b_i
        term = np.divide(numerator, denominator)
        # Handle 0/0 or X/0. if s_i+b_i = 0, separation is 0.
        term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
        sep_density = 0.5 * term

    # Total Separation S
    total_S = np.sum(sep_density)
    
    # Plot Separation Density
    label_text = f'Separation Density (Total S = {total_S:.4f})'
    ax1.plot(bin_centers, sep_density, marker='o', linestyle='-', linewidth=2, color='green', label=label_text)
    ax1.fill_between(bin_centers, sep_density, alpha=0.3, color='green')
    
    ax1.set_ylabel('Separation Density')
    ax1.set_xlabel(variable)
    ax1.set_title(f'Separation Power - {variable} - {region}')
    ax1.grid(True)
    ax1.legend()
    
    return fig

def plot_aggregated_signal_background_comparison(data_list, names, variable, bins, region):
    """
    Create a STACKED plot of All Backgrounds + All Signals.
    NO NORMALIZATION (Raw Yields).
    The histograms are stacked: Backgrounds at the bottom, Signals on top.
    
    Args:
        data_list: List of DataFrames for each process (index 0 is Data)
        names: List of process names
        variable: Physics variable to plot
        bins: Bin edges for histogram
        region: Analysis region
    
    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    bins = np.array(bins)
    
    # Separate and Aggregate Backgrounds and Signals
    
    all_sig_data = []
    all_sig_weights = []
    
    all_bk_data = []
    all_bk_weights = []
    
    for i, (data, name) in enumerate(zip(data_list[1:], names[1:])):
        if "Signal" in name:
            all_sig_data.append(data[variable])
            all_sig_weights.append(data['global_weight'])
        elif "Background" in name:
            all_bk_data.append(data[variable])
            all_bk_weights.append(data['global_weight'])
        else:
            # Fallback
            print(f"Warning: Unknown process type for {name} in Aggregated plot, treating as Background")
            all_bk_data.append(data[variable])
            all_bk_weights.append(data['global_weight'])

    plot_data = [] # List of arrays [bk, sig]
    plot_weights = [] # List of weight arrays
    colors = []
    labels = []

    # 1. Backgrounds (Bottom of stack)
    if all_bk_data:
        bk_concat = np.concatenate(all_bk_data)
        bk_weights_concat = np.concatenate(all_bk_weights)
        
        plot_data.append(bk_concat)
        plot_weights.append(bk_weights_concat)
        colors.append('red')
        labels.append('Total Background')
    else:
        print("Warning: No Background events found")

    # 2. Signals (Top of stack)
    if all_sig_data:
        sig_concat = np.concatenate(all_sig_data)
        sig_weights_concat = np.concatenate(all_sig_weights)
        
        plot_data.append(sig_concat)
        plot_weights.append(sig_weights_concat)
        colors.append('blue')
        labels.append('Total Signal')
    else:
        print("Warning: No Signal events found")

    if plot_data:
        # Plot Stacked, Density=False (Raw counts/yields)
        ax1.hist(plot_data, bins=bins, weights=plot_weights, stacked=True, density=False,
                 histtype='stepfilled', alpha=0.5, color=colors, label=labels, edgecolor='black', linewidth=1)
    else:
        print("Warning: No data to plot")

    ax1.set_ylabel('Events (Raw Yields)')
    ax1.set_xlabel(variable)
    ax1.set_title(f'Stacked Signal vs Background (Raw) - {variable} - {region}')
    ax1.legend()
    ax1.grid(True)
    
    return fig
