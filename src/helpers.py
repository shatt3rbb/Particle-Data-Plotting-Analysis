import pandas as pd
import yaml
import os
import uproot3 as uproot
import numpy as np

def load_run_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['analysis']['region'], config['analysis']['control_type'], config['analysis']['event_type'], config['analysis']['scaling_option']

def get_VARIABLE_BINNING():
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['variable_binning']

def initialize_unscaled_factors():
    #Return scaling factors set to 1.0 (no scaling)
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['scaling_factors']['Unscaled']

def initialize_scaling_factors(scaling_option):
    #Return scaling factors based on the scaling option
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    scaling_factors = config['scaling_factors']
    if scaling_option not in scaling_factors:
        raise SystemExit('The scaling option you have chosen is not correct. Available scaling options are: ' + str(list(scaling_factors.keys())))
    factors = scaling_factors[scaling_option]
    return factors

def get_scaling_factors(scaling_option):
    if(scaling_option == 'Unscaled'):
        return initialize_unscaled_factors()
    else:
        return initialize_scaling_factors(scaling_option)
    
def get_SAMPLES_base_path():
    with open('./src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['BASE_PATH']

def region_and_event_type_check(region, event_type, base_path):
    allowed_event_types = ['all', 'ee', 'mm', 'mmm', 'mme', 'mee', 'eee']
    #Simply checks the folder names under the SAMPLES folder in the base path. Does not actually check if these are valid folders that contain .root files.
    if(region not in os.listdir(base_path)):
        raise SystemExit('The region you have chosen is not correct. Available regions for the existing folder are: ' + str(os.listdir(base_path))[1:-1])
    else:
        base_path += region + '/'
    #Do a check if event_type is correctly input
    if event_type not in allowed_event_types:
        raise SystemExit('The event type you have chosen is not correct. Available event types are: ' + str(allowed_event_types))
    return base_path

# ...existing code...
def get_SAMPLE_PATHS(base_path):

    # List all .root files (non-recursive) and build a mapping:
    files = sorted(
        fname for fname in os.listdir(base_path)
        #AND NOT ignores all the root files starting with FT!!!(or ft)
        if os.path.isfile(os.path.join(base_path, fname)) and fname.lower().endswith('.root') and not fname.upper().startswith('FT') and not fname.startswith('llvvjj_WW')
    )
    # Map sample key (filename without extension) -> filename (relative to base_path)
    sample_map = {os.path.splitext(fname)[0]: fname for fname in files}
    return sample_map

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

# ...existing code...
def apply_scaling_factors(samples, scaling_factors, config_path='./src/config/config.yaml'):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    app = config.get('scaling_application', {})

    # 1) Apply 'no_jets' global scalings
    for sf_key, sample_list in app.get('no_jets', {}).items():
        sf_val = scaling_factors.get(sf_key)
        if sf_val is None:
            # skip unknown scaling key
            continue
        for sample_name in sample_list:
            if sample_name not in samples:
                continue
            # multiply entire global_weight column
            samples[sample_name]['global_weight'] = samples[sample_name]['global_weight'] * sf_val

    # 2) Apply 'with_jets' jet-bin dependent scalings
    for sf_key, sample_list in app.get('with_jets', {}).items():
        # infer channel suffix (ee/mm) from the key (e.g. sf_x_ee -> 'ee')
        chan = None
        if sf_key.endswith('_ee'):
            chan = 'ee'
        elif sf_key.endswith('_mm'):
            chan = 'mm'
        else:
            parts = sf_key.split('_')
            if parts and parts[-1] in ('ee', 'mm'):
                chan = parts[-1]

        if chan is None:
            # unable to infer channel, skip entry
            continue

        # build expected scaling factor keys per jet bin
        bin_sf_keys = {
            0: f"sf_0_{chan}",
            1: f"sf_1_{chan}",
            2: f"sf_2_{chan}",
            'gt2': f"sf_3_{chan}",
        }

        for sample_name in sample_list:
            if sample_name not in samples:
                continue
            df = samples[sample_name]
            # ensure n_jets column exists
            if 'n_jets' not in df.columns or 'global_weight' not in df.columns:
                continue
            # Apply per-bin scaling
            for jet_bin, key in bin_sf_keys.items():
                sf_val = scaling_factors.get(key)
                if sf_val is None:
                    continue
                if jet_bin == 'gt2':
                    mask = df['n_jets'] > 2
                else:
                    mask = df['n_jets'] == jet_bin
                # use loc to preserve alignment
                samples[sample_name].loc[mask, 'global_weight'] = samples[sample_name].loc[mask, 'global_weight'] * sf_val

    return samples

def create_signal_and_background(data, config_path='./src/config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    classification = config.get('classification', {})
    signal = {}
    background = {}
    for key, val in classification['signal'].items():
        try:
            signal[key] = pd.concat(data[sample] for sample in val if sample in data)
        except (KeyError, ValueError):
            Warning = f"Problem when trying to create signal from data. Make sure all your root files are in place and Classification in config.yaml is aligned."
            print(Warning)
            continue
    for key, val in classification['background'].items():
        try:
            background[key] = pd.concat(data[sample] for sample in val if sample in data)
        except (KeyError, ValueError):
            Warning = f"Problem when trying to create background from data. Make sure all your root files are in place and Classification in config.yaml is aligned."
            print(Warning)
            continue
    return signal, background

def get_channels_towards_yields(event_type, region):
    if event_type == 'all':
        if '3lCR' in region:
            channels = ['all', 'mmm', 'mme', 'mee', 'eee']
        else:
            channels = ['all', 'ee', 'mm']
    else:
        channels = [event_type]
    return channels
