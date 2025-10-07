"""
Tests for cmd_verify enhancements (mode detection and verification).

Tests the Phase 3 refactoring which adds:
- Mode parameter support (core/full/auto)
- Auto-detection of file mode
- Mode-aware verification using FeatureVerifier
"""

import pytest
from pathlib import Path
from argparse import Namespace
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the functions to test
from ign_lidar.cli import cmd_verify
from ign_lidar.verification import (
    CORE_FEATURES, 
    FULL_MODE_FEATURES, 
    EXPECTED_FEATURES,
    FeatureVerifier
)


class TestCmdVerifyModeDetection:
    """Test mode detection logic in cmd_verify."""
    
    def test_explicit_core_mode(self, tmp_path, caplog):
        """Test verification with explicitly specified core mode."""
        # Create a test file
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='core',
            quiet=False,
            show_samples=False
        )
        
        # Mock FeatureVerifier to avoid needing real LAZ files
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                result = cmd_verify(args)
        
        # Check that mode was logged as specified
        assert "Mode: CORE (user-specified)" in caplog.text
        
        # Check FeatureVerifier was initialized with core mode
        mock_verifier.assert_called_once()
        call_kwargs = mock_verifier.call_args[1]
        assert call_kwargs['mode'] == 'core'
    
    def test_explicit_full_mode(self, tmp_path, caplog):
        """Test verification with explicitly specified full mode."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='full',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES + FULL_MODE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES + FULL_MODE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                result = cmd_verify(args)
        
        # Check that mode was logged as specified
        assert "Mode: FULL (user-specified)" in caplog.text
        
        # Check FeatureVerifier was initialized with full mode
        mock_verifier.assert_called_once()
        call_kwargs = mock_verifier.call_args[1]
        assert call_kwargs['mode'] == 'full'
    
    def test_auto_detect_core_mode(self, tmp_path, caplog):
        """Test auto-detection of core mode (no building features)."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='auto',
            quiet=False,
            show_samples=False
        )
        
        # Mock laspy to return a file with only core features
        mock_las = Mock()
        mock_las.point_format.extra_dimension_names = [
            'linearity', 'planarity', 'sphericity', 
            'anisotropy', 'roughness', 'density', 'verticality'
        ]
        
        with patch('laspy.open') as mock_open, \
             patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            
            # Setup laspy mock
            mock_context = MagicMock()
            mock_context.__enter__.return_value.read.return_value = mock_las
            mock_open.return_value = mock_context
            
            # Setup verifier mock
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                result = cmd_verify(args)
        
        # Check auto-detection logged correctly
        assert "Mode: AUTO-DETECT" in caplog.text
        assert "Detected CORE mode" in caplog.text
        
        # Check FeatureVerifier was initialized with core mode
        call_kwargs = mock_verifier.call_args[1]
        assert call_kwargs['mode'] == 'core'
    
    def test_auto_detect_full_mode(self, tmp_path, caplog):
        """Test auto-detection of full mode (has building features)."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='auto',
            quiet=False,
            show_samples=False
        )
        
        # Mock laspy to return a file with full mode features
        mock_las = Mock()
        mock_las.point_format.extra_dimension_names = [
            'linearity', 'planarity', 'sphericity', 
            'anisotropy', 'roughness', 'density', 'verticality',
            'wall_score', 'roof_score', 'num_points_2m'
        ]
        
        with patch('laspy.open') as mock_open, \
             patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            
            # Setup laspy mock
            mock_context = MagicMock()
            mock_context.__enter__.return_value.read.return_value = mock_las
            mock_open.return_value = mock_context
            
            # Setup verifier mock
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES + FULL_MODE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES + FULL_MODE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                result = cmd_verify(args)
        
        # Check auto-detection logged correctly
        assert "Mode: AUTO-DETECT" in caplog.text
        assert "Detected FULL mode" in caplog.text
        
        # Check FeatureVerifier was initialized with full mode
        call_kwargs = mock_verifier.call_args[1]
        assert call_kwargs['mode'] == 'full'
    
    def test_auto_detect_fallback_on_error(self, tmp_path, caplog):
        """Test that auto-detection falls back to core mode on error."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='auto',
            quiet=False,
            show_samples=False
        )
        
        # Mock laspy to raise an exception
        with patch('laspy.open') as mock_open, \
             patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            
            mock_open.side_effect = Exception("File read error")
            
            # Setup verifier mock
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.WARNING):
                result = cmd_verify(args)
        
        # Check that fallback was logged
        assert "Auto-detection failed" in caplog.text
        assert "assuming CORE mode" in caplog.text
        
        # Check FeatureVerifier was initialized with core mode (fallback)
        call_kwargs = mock_verifier.call_args[1]
        assert call_kwargs['mode'] == 'core'


class TestCmdVerifyModeLogging:
    """Test that cmd_verify logs appropriate information about modes."""
    
    def test_logs_feature_counts(self, tmp_path, caplog):
        """Test that verification logs correct feature counts for each mode."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        # Test core mode
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='core',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                cmd_verify(args)
        
        # Check feature count logging
        assert f"Checking for {len(CORE_FEATURES)} features" in caplog.text
        assert f"{len(CORE_FEATURES)} core geometric features" in caplog.text
        
        # Should NOT mention building features for core mode
        assert "building-specific" not in caplog.text
    
    def test_logs_building_features_for_full_mode(self, tmp_path, caplog):
        """Test that full mode logs mention building-specific features."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='full',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES + FULL_MODE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES + FULL_MODE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            with caplog.at_level(logging.INFO):
                cmd_verify(args)
        
        # Check that both feature types are mentioned
        assert f"{len(CORE_FEATURES)} core geometric features" in caplog.text
        assert f"{len(FULL_MODE_FEATURES)} full mode features (building-specific)" in caplog.text


class TestCmdVerifyBackwardCompatibility:
    """Test that cmd_verify maintains backward compatibility."""
    
    def test_default_mode_is_auto(self, tmp_path):
        """Test that mode defaults to 'auto' if not specified."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        # Args without mode attribute (simulating old code)
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            quiet=False,
            show_samples=False
        )
        # Don't set args.mode - should default to 'auto'
        
        # Mock laspy
        mock_las = Mock()
        mock_las.point_format.extra_dimension_names = CORE_FEATURES
        
        with patch('laspy.open') as mock_open, \
             patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            
            mock_context = MagicMock()
            mock_context.__enter__.return_value.read.return_value = mock_las
            mock_open.return_value = mock_context
            
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            # Should not raise an error
            result = cmd_verify(args)
        
        # Should have used auto-detection
        assert mock_open.called
        assert mock_verifier.called
    
    def test_verifier_gets_correct_parameters(self, tmp_path):
        """Test that FeatureVerifier receives all expected parameters."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='core',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            cmd_verify(args)
        
        # Check all parameters were passed
        mock_verifier.assert_called_once()
        call_kwargs = mock_verifier.call_args[1]
        
        assert 'mode' in call_kwargs
        assert call_kwargs['mode'] == 'core'
        assert 'sample_size' in call_kwargs
        assert 'check_rgb' in call_kwargs
        assert call_kwargs['check_rgb'] is True
        assert 'check_infrared' in call_kwargs
        assert call_kwargs['check_infrared'] is True


class TestCmdVerifyIntegration:
    """Integration tests for cmd_verify with mode support."""
    
    def test_core_mode_expects_7_features(self, tmp_path):
        """Test that core mode verification expects exactly 7 features."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='core',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.expected_features = CORE_FEATURES
            mock_instance.verify_file.return_value = {
                'present_features': CORE_FEATURES,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            cmd_verify(args)
            
            # Verify the verifier was created with core mode
            call_kwargs = mock_verifier.call_args[1]
            assert call_kwargs['mode'] == 'core'
            
            # Verify expected features count
            assert len(mock_instance.expected_features) == 7
    
    def test_full_mode_expects_10_features(self, tmp_path):
        """Test that full mode verification expects exactly 10 features."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        args = Namespace(
            input=test_file,
            input_dir=None,
            output_dir=None,
            max_files=None,
            mode='full',
            quiet=False,
            show_samples=False
        )
        
        with patch('ign_lidar.cli.FeatureVerifier') as mock_verifier:
            mock_instance = Mock()
            all_features = CORE_FEATURES + FULL_MODE_FEATURES
            mock_instance.expected_features = all_features
            mock_instance.verify_file.return_value = {
                'present_features': all_features,
                'missing_features': [],
                'artifact_features': [],
                'has_rgb': True,
                'has_infrared': False
            }
            mock_verifier.return_value = mock_instance
            
            cmd_verify(args)
            
            # Verify the verifier was created with full mode
            call_kwargs = mock_verifier.call_args[1]
            assert call_kwargs['mode'] == 'full'
            
            # Verify expected features count
            assert len(mock_instance.expected_features) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
