import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import custom_plots

@pytest.fixture
def dummy_data():
    data_df = pd.DataFrame({'pt': [10.0, 20.0], 'global_weight': [1.0, 1.0]})
    # Signal: sum weight = 1.0
    sig_df = pd.DataFrame({'pt': [15.0], 'global_weight': [1.0]})
    # Background: sum weight = 2.0
    bkg_df = pd.DataFrame({'pt': [12.0, 22.0], 'global_weight': [1.0, 1.0]})
    
    data_list = [data_df, sig_df, bkg_df]
    names = ['Data', 'Signal_1', 'Background_1']
    bins = [0, 10, 20, 30]
    
    return data_list, names, bins, 'SR', 'pt'

@patch('src.custom_plots.plt.subplots')
def test_plot_signal_over_background(mock_subplots, dummy_data):
    # Setup mock
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax1)
    
    data_list, names, bins, region, variable = dummy_data
    
    fig = custom_plots.plot_signal_over_background(data_list, names, variable, bins, region)
    
    assert fig == mock_fig
    # Checks if ax1.plot was called (the ratio line)
    assert mock_ax1.plot.called
    
    # We can inspect what it was called with to verify the ratio math if desired
    call_args, call_kwargs = mock_ax1.plot.call_args
    x_centers = call_args[0]
    y_ratios = call_args[1]
    
    # Bin centers for [0, 10, 20, 30] are [5, 15, 25]
    np.testing.assert_allclose(x_centers, [5.0, 15.0, 25.0])
    
    # Signal falls in bin 1 ([10, 20]): weight 1.0. Other bins 0.0.
    # Background falls in bin 1 (12.0 -> weight 1.0), and bin 2 (22.0 -> weight 1.0).
    # expected S/B = [0/0, 1.0/1.0, 0/1.0] -> [0.0, 1.0, 0.0]
    np.testing.assert_allclose(y_ratios, [0.0, 1.0, 0.0])

@patch('src.custom_plots.plt.subplots')
def test_plot_separation_density(mock_subplots, dummy_data):
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax1)
    
    data_list, names, bins, region, variable = dummy_data
    
    fig = custom_plots.plot_separation_density(data_list, names, variable, bins, region)
    
    assert fig == mock_fig
    assert mock_ax1.plot.called
    assert mock_ax1.fill_between.called
    
    call_args, _ = mock_ax1.plot.call_args
    y_sep_dens = call_args[1]
    
    # Sig: [0.0, 1.0, 0.0] -> total=1.0 -> normalized s_i = [0.0, 1.0, 0.0]
    # Bkg: [0.0, 1.0, 1.0] -> total=2.0 -> normalized b_i = [0.0, 0.5, 0.5]
    # Sep_i = 0.5 * (s_i - b_i)^2 / (s_i + b_i)
    # Bin 0: s=0, b=0 -> 0
    # Bin 1: s=1, b=0.5 -> 0.5 * (0.5)^2 / 1.5 = 0.5 * 0.25 / 1.5 = 0.125 / 1.5 = 0.08333...
    # Bin 2: s=0, b=0.5 -> 0.5 * (-0.5)^2 / 0.5 = 0.5 * 0.25 / 0.5 = 0.25
    expected_sep = [0.0, 0.08333333, 0.25]
    
    np.testing.assert_allclose(y_sep_dens, expected_sep, atol=1e-5)

@patch('src.custom_plots.plt.subplots')
def test_plot_aggregated_signal_background_comparison(mock_subplots, dummy_data):
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax1)
    
    data_list, names, bins, region, variable = dummy_data
    
    fig = custom_plots.plot_aggregated_signal_background_comparison(data_list, names, variable, bins, region)
    
    assert fig == mock_fig
    # Should call ax1.hist for drawing stacked plots
    assert mock_ax1.hist.called
    
    call_args, call_kwargs = mock_ax1.hist.call_args
    plot_data = call_args[0]  # list of arrays [bk, sig]
    
    assert len(plot_data) == 2
    # bk data
    np.testing.assert_array_equal(plot_data[0], [12.0, 22.0])
    # sig data
    np.testing.assert_array_equal(plot_data[1], [15.0])
