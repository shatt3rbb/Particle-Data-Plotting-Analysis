import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import bdt_analysis

@pytest.fixture
def dummy_bdt_data():
    features = ['met_tst', 'MetOHT', 'dLepR', 'dPhill', 'dMetZPhi']
    
    # Create a dummy DataFrame with 100 rows
    np.random.seed(42)
    sig_data = {feat: np.random.normal(1.0, 0.5, 50) for feat in features}
    sig_data['global_weight'] = np.ones(50) * 2.0
    sig_df = pd.DataFrame(sig_data)
    
    bkg_data = {feat: np.random.normal(0.0, 0.5, 50) for feat in features}
    bkg_data['global_weight'] = np.ones(50) * 1.0
    bkg_df = pd.DataFrame(bkg_data)
    
    # We mock helpers.create_signal_and_background to return dictionaries
    return {'sig1': sig_df}, {'bkg1': bkg_df}

@patch('src.bdt_analysis.plt')
def test_optimize_bdt_cut(mock_plt):
    # Mocking a classifier with a deterministic decision function return
    clf_mock = MagicMock()
    # Scores: -2, -1, 0, 1, 2
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    clf_mock.decision_function.return_value = scores
    
    X_mock = np.zeros((5, 5)) 
    # y = [B, B, B, S, S]
    y_test = np.array([0, 0, 0, 1, 1])
    # weights = [1, 1, 1, 1, 1]
    w_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Since cuts scan 100 points between -2 to 2, we expect a cut at > 0.0
    # to yield S=2, B=0, which maximizes significance.
    # Actually, B=0 gives significance 0 by our formula logic, 
    # so it will pick cut > -1.0 which gives S=2, B=1
    
    # We call it
    bdt_analysis.optimize_bdt_cut(clf_mock, X_mock, y_test, w_test, 'dummy_dir', scale_factor=1.0)
    
    # Should save the significance plot
    assert mock_plt.savefig.called
    assert mock_plt.close.called

@patch('src.bdt_analysis.helpers')
@patch('src.bdt_analysis.plt')
@patch('src.bdt_analysis.os.makedirs')
def test_run_bdt_analysis(mock_makedirs, mock_plt, mock_helpers, dummy_bdt_data):
    # 1. Setup mocks to bypass ROOT file loading
    mock_helpers.load_run_config.return_value = ('SR', 'typeA', 'all', 'nominal')
    mock_helpers.get_SAMPLES_base_path.return_value = '/fake/'
    mock_helpers.region_and_event_type_check.return_value = '/fake/SR/'
    mock_helpers.get_SAMPLE_PATHS.return_value = {'sample1': 'file1.root'}
    
    # The crucial part: mock the load/process chain so we don't open real files
    # Actually, since get_SAMPLE_PATHS has 1 file, it loops once.
    mock_helpers.load_data_filtering_event_type.return_value = pd.DataFrame() # Doesn't matter, overridden below
    mock_helpers.calculate_derived_variables.return_value = pd.DataFrame()
    mock_helpers.apply_analysis_cuts.return_value = pd.DataFrame()
    mock_helpers.get_scaling_factors.return_value = {}
    mock_helpers.apply_scaling_factors.return_value = {}
    
    # Provide the dummy dataframes directly
    sig_dict, bkg_dict = dummy_bdt_data
    mock_helpers.create_signal_and_background.return_value = (sig_dict, bkg_dict)
    
    # 2. Call the main function
    bdt_analysis.run_bdt_analysis()
    
    # 3. Asserts
    assert mock_makedirs.called
    # It generates 4 plots: roc_curve, score_distribution, feature_importance, score_distribution_log_yield
    # plus 1 from significance scan
    assert mock_plt.savefig.call_count >= 4
    assert mock_plt.close.call_count >= 4
