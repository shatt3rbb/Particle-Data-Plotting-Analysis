import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import calculators

def test_calculate_significance():
    # Regular case: S=10, B=90
    sig = calculators.calculate_significance(10, 90)
    expected = np.sqrt(2 * ((10 + 90) * np.log(1 + (10 / 90)) - 10))
    np.testing.assert_allclose(sig, expected)

    # Division by zero or value error case
    assert calculators.calculate_significance(10, 0) == 0
    assert calculators.calculate_significance(-10, -90) == 0

def test_calculate_yields():
    data = pd.DataFrame({
        'event_type': [1, 0, 1, 0],
        'event_3CR': [1, 2, 3, 4],
        'global_weight': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Test 'ee' (event_type == 1)
    sig, unc = calculators.calculate_yields(data, 'ee')
    assert sig == 4.0 # 1.0 + 3.0
    np.testing.assert_allclose(unc, np.sqrt(1.0**2 + 3.0**2))
    
    # Test 'mm' (event_type == 0)
    sig, unc = calculators.calculate_yields(data, 'mm')
    assert sig == 6.0 # 2.0 + 4.0
    np.testing.assert_allclose(unc, np.sqrt(2.0**2 + 4.0**2))
    
    # Test 'mmm' (event_3CR == 1)
    sig, unc = calculators.calculate_yields(data, 'mmm')
    assert sig == 1.0
    np.testing.assert_allclose(unc, 1.0)

def test_calculate_all_yields():
    data_df = pd.DataFrame({'event_type': [1, 1], 'global_weight': [10.0, 10.0]})
    sig_df = pd.DataFrame({'event_type': [1], 'global_weight': [5.0]})
    bkg_df = pd.DataFrame({'event_type': [1, 1], 'global_weight': [100.0, 200.0]})
    
    signal = {'sig1': sig_df}
    background = {'bkg1': bkg_df}
    channels = ['ee']
    
    results = calculators.calculate_all_yields(data_df, signal, background, channels)
    
    assert 'ee' in results
    ch_result = results['ee']
    
    # Data yields
    assert ch_result['data'][0] == 20.0
    np.testing.assert_allclose(ch_result['data'][1], np.sqrt(10.0**2 + 10.0**2))
    
    # Signal yields
    assert ch_result['signals']['sig1'][0] == 5.0
    
    # Background yields
    assert ch_result['backgrounds']['bkg1'][0] == 300.0
    np.testing.assert_allclose(ch_result['backgrounds']['bkg1'][1], np.sqrt(100.0**2 + 200.0**2))
    
    # Total signal and background
    assert ch_result['total_signal'][0] == 5.0
    assert ch_result['total_bkg'][0] == 300.0
    
    # Significance
    expected_sig = calculators.calculate_significance(5.0, 300.0)
    assert ch_result['significance'] == expected_sig

@patch('src.calculators.plt.subplots')
def test_plot_distributions(mock_subplots):
    # Setup mock
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
    
    data_df = pd.DataFrame({'pt': [10, 20, 30], 'global_weight': [1.0, 1.0, 1.0]})
    mc_df = pd.DataFrame({'pt': [15, 25], 'global_weight': [0.5, 0.5]})
    
    data_list = [data_df, mc_df]
    names = ['Data', 'MC']
    bins = [0, 10, 20, 30, 40]
    
    fig = calculators.plot_distributions(data_list, names, 'pt', bins, 'SR')
    
    assert fig == mock_fig
    # Should call hist for MC
    assert mock_ax1.hist.called
    # Should call errorbar for data and ratio
    assert mock_ax1.errorbar.called
    assert mock_ax2.errorbar.called

def test_create_yield_table():
    results = {
        'ee': {
            'data': (100.0, 10.0),
            'signals': {'sig1': (10.0, 1.0)},
            'backgrounds': {'bkg1': (90.0, 9.0)},
            'total_signal': (10.0, 1.0),
            'total_bkg': (90.0, 9.0),
            'significance': 1.0
        }
    }
    
    df = calculators.create_yield_table(results, ['ee'])
    
    assert list(df.columns) == ['Process', 'Yield', 'Uncertainty']
    # Check data row
    assert 'Data (ee)' in df['Process'].values
    # Check significance
    assert 'Significance (ee)' in df['Process'].values

@patch('src.calculators.os.makedirs')
@patch('src.calculators.plt.close')
def test_save_results(mock_close, mock_makedirs):
    with patch('src.calculators.create_yield_table') as mock_create_table:
        mock_df = MagicMock()
        mock_create_table.return_value = mock_df
        
        mock_fig = MagicMock()
        
        results = {'ee': {'significance': 1.5}}
        
        with patch('pandas.DataFrame.to_csv'), \
             patch('builtins.open', new_callable=mock_open):
            calculators.save_results(
                results=results, 
                figures=[mock_fig], 
                channels=['ee'], 
                region='SR', 
                control_type='typeA', 
                event_type='all', 
                scaling_option='nominal', 
                variable_binning={'pt': [0, 10]}
            )
            
            mock_makedirs.assert_called_once()
            mock_df.to_csv.assert_called_once()
            mock_fig.savefig.assert_called_once()
            mock_close.assert_called_once_with(mock_fig)
