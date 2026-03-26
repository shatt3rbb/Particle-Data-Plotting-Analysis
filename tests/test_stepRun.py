import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import stepRun

@pytest.fixture
def dummy_steprun_data():
    data_df = pd.DataFrame({'pt': [10.0, 20.0], 'global_weight': [1.0, 1.0]})
    # Signal: sum weight = 1.0
    sig_df = pd.DataFrame({'pt': [15.0], 'global_weight': [1.0]})
    # Background: sum weight = 2.0
    bkg_df = pd.DataFrame({'pt': [12.0, 22.0], 'global_weight': [1.0, 1.0]})
    
    return data_df, sig_df, bkg_df

@patch('src.stepRun.helpers')
@patch('src.stepRun.calculators')
@patch('src.stepRun.custom_plots')
def test_steprun_main(mock_custom_plots, mock_calculators, mock_helpers, dummy_steprun_data):
    # Retrieve dummy data
    data_df, sig_df, bkg_df = dummy_steprun_data
    
    # Setup Mocks for helpers
    mock_helpers.load_run_config.return_value = ('SR', 'typeA', 'all', 'nominal')
    mock_helpers.get_scaling_factors.return_value = {}
    mock_helpers.get_SAMPLES_base_path.return_value = '/fake/'
    mock_helpers.region_and_event_type_check.return_value = '/fake/SR/'
    
    # Let's say we have 1 file to process
    mock_helpers.get_SAMPLE_PATHS.return_value = {'sample1': 'file1.root'}
    
    # Process loop mocks
    mock_helpers.load_data_filtering_event_type.return_value = pd.DataFrame()
    mock_helpers.calculate_derived_variables.return_value = pd.DataFrame()
    mock_helpers.apply_analysis_cuts.return_value = pd.DataFrame()
    
    # Return dummy objects for downstream analysis
    mock_helpers.apply_scaling_factors.return_value = {
        'DATA': data_df,
        'sig1': sig_df,
        'bkg1': bkg_df
    }
    mock_helpers.create_signal_and_background.return_value = (
        {'sig1': sig_df}, 
        {'bkg1': bkg_df}
    )
    mock_helpers.get_channels_towards_yields.return_value = ['ee']
    mock_helpers.get_VARIABLE_BINNING.return_value = {'pt': [0, 10, 20, 30]}
    
    # Setup Mocks for calculators
    yield_results = {'ee': {'significance': 1.5}}
    mock_calculators.calculate_all_yields.return_value = yield_results
    
    # Setup Mocks for custom_plots
    mock_fig = MagicMock()
    mock_custom_plots.plot_aggregated_signal_background_comparison.return_value = mock_fig
    
    # Execute the main function since we refactored it
    stepRun.main()
    
    # Assertions
    assert mock_helpers.load_run_config.called
    assert mock_helpers.get_SAMPLE_PATHS.called
    assert mock_calculators.calculate_all_yields.called
    assert mock_custom_plots.plot_aggregated_signal_background_comparison.called
    assert mock_calculators.save_results.called
