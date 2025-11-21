"""
Test suite for multi-scale GPU connection improvements.

Tests the enhanced GPU validation, error handling, and fallback logic.

**IMPORTANT: Run with ign_gpu conda environment:**

    conda run -n ign_gpu python -m pytest tests/test_multi_scale_gpu_connection.py -v

"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock


class TestMultiScaleGPUConnection:
    """Test multi-scale GPU connection functionality."""
    
    def create_orchestrator_mock(self, use_gpu=False):
        """Create a mock orchestrator with minimal setup."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator
        
        # Create mock with just the method we're testing
        orchestrator = Mock(spec=FeatureOrchestrator)
        orchestrator.config = {
            "features": {
                "use_gpu": use_gpu,
                "force_gpu": False
            }
        }
        orchestrator.use_multi_scale = True
        orchestrator.multi_scale_computer = Mock()
        
        # Bind the actual method to the mock
        from types import MethodType
        orchestrator._connect_multi_scale_gpu = MethodType(
            FeatureOrchestrator._connect_multi_scale_gpu,
            orchestrator
        )
        
        return orchestrator
    
    def test_no_connection_when_gpu_not_requested(self, caplog):
        """Test that GPU connection is skipped when not requested."""
        caplog.set_level(logging.DEBUG)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=False)
        orchestrator.computer = Mock()
        
        orchestrator._connect_multi_scale_gpu()
        
        # Should log that CPU is being used
        assert any('Multi-scale using CPU' in msg for msg in caplog.messages)
    
    def test_skip_when_multi_scale_disabled(self):
        """Test that connection is skipped when multi-scale is disabled."""
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        orchestrator.use_multi_scale = False
        orchestrator.multi_scale_computer = None
        
        # Should return early without errors
        orchestrator._connect_multi_scale_gpu()
    
    def test_successful_gpu_connection(self, caplog):
        """Test successful GPU connection with validation."""
        caplog.set_level(logging.INFO)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        
        # Mock computer with GPU
        mock_gpu_processor = Mock()
        orchestrator.computer = Mock()
        orchestrator.computer.gpu_processor = mock_gpu_processor
        
        # Mock CuPy
        with patch('ign_lidar.features.orchestrator.cp') as mock_cp:
            # Mock GPU device info
            mock_device = Mock()
            mock_device.name = "NVIDIA RTX 3080"
            mock_device.mem_info = (8 * 1e9, 10 * 1e9)  # 8GB free, 10GB total
            mock_cp.cuda.Device.return_value = mock_device
            
            # Mock GPU test
            mock_cp.array.return_value = Mock()
            mock_cp.mean.return_value = 2.0
            
            orchestrator._connect_multi_scale_gpu()
            
            # Verify GPU was connected
            assert orchestrator.multi_scale_computer.use_gpu is True
            assert orchestrator.multi_scale_computer.gpu_processor is not None
            
            # Check success message
            assert any('Multi-scale connected to GPU' in msg for msg in caplog.messages)
            assert any('NVIDIA RTX 3080' in msg for msg in caplog.messages)
    
    def test_gpu_not_available_warning(self, caplog):
        """Test warning when GPU processor not available."""
        caplog.set_level(logging.WARNING)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        orchestrator.computer = Mock(spec=[])  # No gpu_processor attribute
        
        orchestrator._connect_multi_scale_gpu()
        
        # Should fall back to CPU
        assert orchestrator.multi_scale_computer.use_gpu is False
        
        # Check warning messages
        log_text = ' '.join(caplog.messages)
        assert 'GPU requested but no GPU processor available' in log_text
        assert 'CuPy is not installed' in log_text
        assert 'falling back to CPU' in log_text
    
    def test_cupy_not_installed_fallback(self, caplog):
        """Test fallback when CuPy is not installed."""
        caplog.set_level(logging.ERROR)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        
        mock_gpu_processor = Mock()
        orchestrator.computer = Mock()
        orchestrator.computer.gpu_processor = mock_gpu_processor
        
        # Mock CuPy import failure
        with patch('ign_lidar.features.orchestrator.cp', side_effect=ImportError("No module named 'cupy'")):
            orchestrator._connect_multi_scale_gpu()
            
            # Should fall back to CPU
            assert orchestrator.multi_scale_computer.use_gpu is False
            
            # Check error messages
            log_text = ' '.join(caplog.messages)
            assert 'CuPy not installed' in log_text
            assert 'conda install' in log_text or 'pip install' in log_text
            assert 'falling back to CPU' in log_text
    
    def test_gpu_validation_failure_fallback(self, caplog):
        """Test fallback when GPU validation fails."""
        caplog.set_level(logging.ERROR)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        
        mock_gpu_processor = Mock()
        orchestrator.computer = Mock()
        orchestrator.computer.gpu_processor = mock_gpu_processor
        
        # Mock CuPy with GPU error
        with patch('ign_lidar.features.orchestrator.cp') as mock_cp:
            mock_cp.array.side_effect = Exception("CUDA out of memory")
            
            orchestrator._connect_multi_scale_gpu()
            
            # Should fall back to CPU
            assert orchestrator.multi_scale_computer.use_gpu is False
            
            # Check error messages
            log_text = ' '.join(caplog.messages)
            assert 'GPU validation failed' in log_text
            assert 'CUDA out of memory' in log_text
            assert 'falling back to CPU' in log_text
    
    def test_actionable_error_messages(self, caplog):
        """Test that error messages provide actionable solutions."""
        caplog.set_level(logging.WARNING)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        orchestrator.computer = Mock(spec=[])  # No GPU
        
        orchestrator._connect_multi_scale_gpu()
        
        log_text = ' '.join(caplog.messages).lower()
        
        # Check for actionable information
        assert any(keyword in log_text for keyword in [
            'install', 'cupy', 'cuda', 'detected'
        ])
        assert 'may occur if' in log_text
    
    def test_gpu_memory_info_logging(self, caplog):
        """Test that GPU memory information is logged."""
        caplog.set_level(logging.INFO)
        
        orchestrator = self.create_orchestrator_mock(use_gpu=True)
        
        mock_gpu_processor = Mock()
        orchestrator.computer = Mock()
        orchestrator.computer.gpu_processor = mock_gpu_processor
        
        with patch('ign_lidar.features.orchestrator.cp') as mock_cp:
            # Mock GPU with 12GB total, 10GB free
            mock_device = Mock()
            mock_device.name = "NVIDIA A100"
            mock_device.mem_info = (10 * 1e9, 12 * 1e9)
            mock_cp.cuda.Device.return_value = mock_device
            mock_cp.array.return_value = Mock()
            mock_cp.mean.return_value = 2.0
            
            orchestrator._connect_multi_scale_gpu()
            
            log_text = ' '.join(caplog.messages)
            
            # Check memory info is logged
            assert 'GPU memory' in log_text
            assert 'GB' in log_text
            # Should show something like "10.0/12.0 GB"
    
    def test_force_gpu_config(self):
        """Test that force_gpu config option works."""
        orchestrator = self.create_orchestrator_mock(use_gpu=False)
        orchestrator.config["features"]["force_gpu"] = True
        
        mock_gpu_processor = Mock()
        orchestrator.computer = Mock()
        orchestrator.computer.gpu_processor = mock_gpu_processor
        
        with patch('ign_lidar.features.orchestrator.cp') as mock_cp:
            mock_device = Mock()
            mock_device.name = "GPU"
            mock_device.mem_info = (8 * 1e9, 10 * 1e9)
            mock_cp.cuda.Device.return_value = mock_device
            mock_cp.array.return_value = Mock()
            mock_cp.mean.return_value = 2.0
            
            orchestrator._connect_multi_scale_gpu()
            
            # GPU should still be used despite use_gpu=False
            assert orchestrator.multi_scale_computer.use_gpu is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

