import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, mock_open

# Ensure we can import the src module correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import helpers

def test_load_run_config():
    mock_yaml_data = {
        'analysis': {
            'region': 'SR',
            'control_type': 'typeA',
            'event_type': 'all',
            'scaling_option': 'nominal'
        }
    }
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        region, control_type, event_type, scaling_option = helpers.load_run_config('dummy_path.yaml')
        assert region == 'SR'
        assert control_type == 'typeA'
        assert event_type == 'all'
        assert scaling_option == 'nominal'

def test_get_VARIABLE_BINNING():
    mock_yaml_data = {'variable_binning': {'pt': [0, 10, 20]}}
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        assert helpers.get_VARIABLE_BINNING() == {'pt': [0, 10, 20]}

def test_initialize_unscaled_factors():
    mock_yaml_data = {'scaling_factors': {'Unscaled': {'factor1': 1.0}}}
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        assert helpers.initialize_unscaled_factors() == {'factor1': 1.0}

def test_initialize_scaling_factors():
    mock_yaml_data = {'scaling_factors': {'nominal': {'factor1': 1.2}}}
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        assert helpers.initialize_scaling_factors('nominal') == {'factor1': 1.2}

def test_initialize_scaling_factors_invalid():
    mock_yaml_data = {'scaling_factors': {'nominal': {'factor1': 1.2}}}
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        with pytest.raises(SystemExit):
            helpers.initialize_scaling_factors('invalid_option')

def test_get_scaling_factors():
    mock_yaml_data = {
        'scaling_factors': {
            'Unscaled': {'factor': 1.0},
            'nominal': {'factor': 1.5}
        }
    }
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        assert helpers.get_scaling_factors('Unscaled') == {'factor': 1.0}
        assert helpers.get_scaling_factors('nominal') == {'factor': 1.5}

def test_region_and_event_type_check():
    with patch('src.helpers.os.listdir', return_value=['SR', 'CR']):
        # Valid case
        base_path = helpers.region_and_event_type_check('SR', 'mm', '/fake/path/')
        assert base_path == '/fake/path/SR/'
        
        # Invalid region
        with pytest.raises(SystemExit):
            helpers.region_and_event_type_check('invalid_region', 'mm', '/fake/path/')
            
        # Invalid event type
        with pytest.raises(SystemExit):
            helpers.region_and_event_type_check('SR', 'invalid_type', '/fake/path/')

def test_calculate_derived_variables():
    # Prepare dummy data
    data = pd.DataFrame({
        'lepplus_phi': [-1.0, 1.0],
        'lepminus_phi': [-0.5, 0.5],
        'met_tst': [100.0, 200.0],
        'Z_pT': [50.0, 100.0],
        'leading_jet_pt': [30.0, 40.0],
        'second_jet_pt': [20.0, 30.0],
        'leading_jet_eta': [1.5, -1.5],
        'second_jet_eta': [1.0, 1.0],
        'dMetZPhi': [0.0, np.pi/2]
    })
    
    # Needs copy here in case function modifies in place
    result = helpers.calculate_derived_variables(data.copy())
    
    # Check lepplus_phi adjustments (< 0 -> + pi)
    np.testing.assert_allclose(result['lepplus_phi'].iloc[0], -1.0 + np.pi)
    np.testing.assert_allclose(result['lepplus_phi'].iloc[1], 1.0)
    
    # dPhill
    expected_dPhill_0 = (-1.0 + np.pi) - (-0.5 + np.pi)
    np.testing.assert_allclose(result['dPhill'].iloc[0], expected_dPhill_0)
    
    # MetOZPt
    np.testing.assert_allclose(result['MetOZPt'].iloc[0], 100.0 / 50.0)
    
    # MetOHT_2
    np.testing.assert_allclose(result['MetOHT_2'].iloc[0], 100.0 / (50.0 + 30.0 + 20.0))
    
    # proshmo (eta signs)
    assert result['proshmo'].iloc[0] == 1.0   # 1.5 * 1.0 > 0
    assert result['proshmo'].iloc[1] == -1.0  # -1.5 * 1.0 < 0
    
    # mT_ZZ
    # Row 0: dMetZPhi = 0 -> cos(0) = 1 -> mT_ZZ = 0
    np.testing.assert_allclose(result['mT_ZZ'].iloc[0], 0.0, atol=1e-7)
    # Row 1: dMetZPhi = pi/2 -> cos(pi/2) = 0 -> mT_ZZ = sqrt(2 * 100 * 200) = 200
    np.testing.assert_allclose(result['mT_ZZ'].iloc[1], 200.0, atol=1e-7)

def test_apply_analysis_cuts():
    mock_yaml_data = {
        'cuts': {
            'CR': {
                'var1': 'x > 10',
                'var2': 'x < 50'
            }
        }
    }
    
    data = pd.DataFrame({
        'var1': [5, 15, 20],
        'var2': [20, 30, 60]
    })
    
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        result = helpers.apply_analysis_cuts(data, 'CR')
        
    # Row 0 fails var1 > 10
    # Row 2 fails var2 < 50
    # Only Row 1 passes both
    assert len(result) == 1
    assert result['var1'].iloc[0] == 15

def test_apply_scaling_factors():
    samples = {
        'sampleA': pd.DataFrame({'global_weight': [1.0, 2.0], 'n_jets': [0, 1]}),
        'sampleB': pd.DataFrame({'global_weight': [1.0, 1.0], 'n_jets': [3, 2]})
    }
    scaling_factors = {
        'sf_no_jets_sampleA': 2.0,
        'sf_0_ee': 1.1,
        'sf_1_ee': 1.2,
        'sf_2_ee': 1.3,
        'sf_3_ee': 1.4,
        'sf_with_jets_ee': 1.0 
    }
    mock_yaml_data = {
        'scaling_application': {
            'no_jets': {
                'sf_no_jets_sampleA': ['sampleA']
            },
            'with_jets': {
                'sf_with_jets_ee': ['sampleB'] # Will apply sf_X_ee based on n_jets for sampleB
            }
        }
    }
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        result = helpers.apply_scaling_factors(samples, scaling_factors, 'dummy.yaml')
        
    # sampleA: global_weight * 2.0
    np.testing.assert_allclose(result['sampleA']['global_weight'].tolist(), [2.0, 4.0])
    
    # sampleB: n_jets=3 -> sf_3_ee (1.4); n_jets=2 -> sf_2_ee (1.3)
    np.testing.assert_allclose(result['sampleB']['global_weight'].tolist(), [1.4, 1.3])

def test_create_signal_and_background():
    data = {
        'sampleA': pd.DataFrame({'val': [1, 2]}),
        'sampleB': pd.DataFrame({'val': [3, 4]}),
        'sampleC': pd.DataFrame({'val': [5, 6]})
    }
    
    mock_yaml_data = {
        'classification': {
            'signal': {'sig_group': ['sampleA']},
            'background': {'bkg_group': ['sampleB', 'sampleC', 'missing_sample']}
        }
    }
    
    with patch('builtins.open', mock_open()), \
         patch('src.helpers.yaml.safe_load', return_value=mock_yaml_data):
        signal, background = helpers.create_signal_and_background(data, 'dummy.yaml')
        
    assert 'sig_group' in signal
    assert len(signal['sig_group']) == 2
    assert signal['sig_group']['val'].tolist() == [1, 2]
    
    assert 'bkg_group' in background
    assert len(background['bkg_group']) == 4
    assert background['bkg_group']['val'].tolist() == [3, 4, 5, 6]

def test_get_channels_towards_yields():
    assert helpers.get_channels_towards_yields('all', 'my_3lCR_region') == ['all', 'mmm', 'mme', 'mee', 'eee']
    assert helpers.get_channels_towards_yields('all', 'SR') == ['all', 'ee', 'mm']
    assert helpers.get_channels_towards_yields('ee', 'SR') == ['ee']


def test_load_data_filtering_event_type():
    from unittest.mock import MagicMock
    
    with patch('src.helpers.uproot.open') as mock_uproot_open:
        # Create a mock tree and file
        mock_file = MagicMock()
        mock_tree = MagicMock()
        mock_file.__getitem__.return_value = mock_tree
        mock_uproot_open.return_value = mock_file
        mock_tree.num_entries = 4

        # Simulate the tree.iterate behavior returning pandas DataFrames
        df1 = pd.DataFrame({'event_type': [0, 1], 'event_3CR': [1, 2], 'other_data': [100, 200]})
        df2 = pd.DataFrame({'event_type': [1, 0], 'event_3CR': [3, 4], 'other_data': [300, 400]})
        
        # Test 'ee' (event_type == 1)
        mock_tree.iterate.return_value = [df1, df2]
        result_ee = helpers.load_data_filtering_event_type('dummy.root', 'ee')
        assert len(result_ee) == 2
        assert result_ee['other_data'].tolist() == [200, 300]
        
        # Test 'mm' (event_type == 0)
        mock_tree.iterate.return_value = [df1, df2]
        result_mm = helpers.load_data_filtering_event_type('dummy.root', 'mm')
        assert len(result_mm) == 2
        assert result_mm['other_data'].tolist() == [100, 400]

        # Test 'mmm' (event_3CR == 1)
        mock_tree.iterate.return_value = [df1, df2]
        result_mmm = helpers.load_data_filtering_event_type('dummy.root', 'mmm')
        assert len(result_mmm) == 1
        assert result_mmm['other_data'].tolist() == [100]
        
        # Test 'all' (No filtering)
        mock_tree.iterate.return_value = [df1, df2]
        result_all = helpers.load_data_filtering_event_type('dummy.root', 'all')
        assert len(result_all) == 4
        assert result_all['other_data'].tolist() == [100, 200, 300, 400]

