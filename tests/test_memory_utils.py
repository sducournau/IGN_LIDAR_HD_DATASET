"""
Tests for memory management utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from ign_lidar.memory_utils import (
    get_system_memory_info,
    estimate_memory_per_worker,
    calculate_optimal_workers,
    calculate_batch_size,
    analyze_file_sizes,
    sort_files_by_size,
    log_memory_configuration
)


class TestSystemMemoryInfo:
    """Tests for get_system_memory_info()."""
    
    @patch('ign_lidar.memory_utils.PSUTIL_AVAILABLE', True)
    @patch('ign_lidar.memory_utils.psutil')
    def test_with_psutil(self, mock_psutil):
        """Test memory info retrieval with psutil available."""
        # Mock memory objects
        mock_mem = Mock()
        mock_mem.available = 8 * 1024**3  # 8GB
        mock_mem.total = 16 * 1024**3     # 16GB
        mock_mem.percent = 50.0
        
        mock_swap = Mock()
        mock_swap.percent = 10.0
        
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_psutil.swap_memory.return_value = mock_swap
        
        info = get_system_memory_info()
        
        assert info['available_gb'] == pytest.approx(8.0, rel=0.1)
        assert info['total_gb'] == pytest.approx(16.0, rel=0.1)
        assert info['percent_used'] == 50.0
        assert info['swap_percent'] == 10.0
        assert info['has_pressure'] is False
    
    @patch('ign_lidar.memory_utils.PSUTIL_AVAILABLE', True)
    @patch('ign_lidar.memory_utils.psutil')
    def test_with_memory_pressure(self, mock_psutil):
        """Test detection of memory pressure."""
        mock_mem = Mock()
        mock_mem.available = 2 * 1024**3  # 2GB
        mock_mem.total = 16 * 1024**3
        mock_mem.percent = 90.0  # High usage
        
        mock_swap = Mock()
        mock_swap.percent = 60.0  # High swap usage
        
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_psutil.swap_memory.return_value = mock_swap
        
        info = get_system_memory_info()
        
        assert info['has_pressure'] is True
    
    @patch('ign_lidar.memory_utils.PSUTIL_AVAILABLE', False)
    def test_without_psutil(self):
        """Test fallback when psutil is not available."""
        info = get_system_memory_info()
        
        assert 'available_gb' in info
        assert 'total_gb' in info
        assert info['available_gb'] > 0
        assert info['has_pressure'] is False


class TestMemoryEstimation:
    """Tests for estimate_memory_per_worker()."""
    
    def test_core_mode_small_file(self):
        """Test memory estimation for small file in core mode."""
        mem_gb = estimate_memory_per_worker(100, mode='core', use_gpu=False)
        
        # Should be relatively low (2-4GB range)
        assert 2.0 <= mem_gb <= 5.0
    
    def test_full_mode_large_file(self):
        """Test memory estimation for large file in full mode."""
        mem_gb = estimate_memory_per_worker(500, mode='full', use_gpu=False)
        
        # Should be higher for full mode
        assert mem_gb > 3.0
    
    def test_gpu_overhead(self):
        """Test GPU adds overhead to memory estimation."""
        mem_cpu = estimate_memory_per_worker(200, mode='core', use_gpu=False)
        mem_gpu = estimate_memory_per_worker(200, mode='core', use_gpu=True)
        
        assert mem_gpu > mem_cpu
    
    def test_minimum_memory(self):
        """Test minimum memory requirement is enforced."""
        mem_gb = estimate_memory_per_worker(10, mode='core', use_gpu=False)
        
        # Should never go below 2GB
        assert mem_gb >= 2.0


class TestOptimalWorkers:
    """Tests for calculate_optimal_workers()."""
    
    @patch('ign_lidar.memory_utils.get_system_memory_info')
    def test_basic_calculation(self, mock_mem_info):
        """Test basic worker calculation."""
        mock_mem_info.return_value = {
            'available_gb': 16.0,
            'total_gb': 32.0,
            'percent_used': 50.0,
            'swap_percent': 10.0,
            'has_pressure': False
        }
        
        workers, info = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[100, 150, 200],
            mode='core',
            use_gpu=False
        )
        
        assert workers >= 1
        assert 'memory_per_worker_gb' in info
        assert 'max_file_size_mb' in info
        assert info['max_file_size_mb'] == 200
    
    @patch('ign_lidar.memory_utils.get_system_memory_info')
    def test_gpu_forces_single_worker(self, mock_mem_info):
        """Test GPU mode forces single worker."""
        mock_mem_info.return_value = {
            'available_gb': 32.0,
            'total_gb': 64.0,
            'percent_used': 30.0,
            'swap_percent': 0.0,
            'has_pressure': False
        }
        
        workers, info = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[100],
            mode='core',
            use_gpu=True,
            requested_workers=8
        )
        
        assert workers == 1
        assert 'CUDA' in info['recommendation_reason'] or 'GPU' in info['recommendation_reason']
    
    @patch('ign_lidar.memory_utils.get_system_memory_info')
    def test_memory_pressure_reduces_workers(self, mock_mem_info):
        """Test memory pressure forces single worker."""
        mock_mem_info.return_value = {
            'available_gb': 2.0,
            'total_gb': 16.0,
            'percent_used': 90.0,
            'swap_percent': 70.0,
            'has_pressure': True
        }
        
        workers, info = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[100],
            mode='core',
            use_gpu=False,
            requested_workers=8
        )
        
        assert workers == 1
        assert info['has_memory_pressure'] is True
    
    @patch('ign_lidar.memory_utils.get_system_memory_info')
    def test_large_files_limit_workers(self, mock_mem_info):
        """Test large files limit worker count."""
        mock_mem_info.return_value = {
            'available_gb': 32.0,
            'total_gb': 64.0,
            'percent_used': 30.0,
            'swap_percent': 0.0,
            'has_pressure': False
        }
        
        workers, info = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[600, 700, 800],  # Very large files
            mode='full',
            use_gpu=False,
            requested_workers=8
        )
        
        # Should limit workers for large files
        assert workers <= 4
        assert 'large' in info['recommendation_reason'].lower()


class TestBatchSize:
    """Tests for calculate_batch_size()."""
    
    def test_core_mode_small_files(self):
        """Test batch size for core mode with small files."""
        batch_size = calculate_batch_size(
            num_workers=4,
            max_file_size_mb=100,
            mode='core'
        )
        
        # Small files can have larger batches
        assert batch_size >= 4
    
    def test_full_mode_large_files(self):
        """Test batch size for full mode with large files."""
        batch_size = calculate_batch_size(
            num_workers=4,
            max_file_size_mb=400,
            mode='full'
        )
        
        # Large files in full mode need sequential processing
        assert batch_size <= 2
    
    def test_minimum_batch_size(self):
        """Test batch size is never less than 1."""
        batch_size = calculate_batch_size(
            num_workers=1,
            max_file_size_mb=1000,
            mode='full'
        )
        
        assert batch_size >= 1


class TestFileSizeAnalysis:
    """Tests for file size analysis functions."""
    
    def test_analyze_file_sizes(self, tmp_path):
        """Test file size analysis."""
        # Create test files
        files = []
        sizes_mb = [10, 20, 30]
        
        for i, size_mb in enumerate(sizes_mb):
            file = tmp_path / f"test_{i}.laz"
            file.write_bytes(b'x' * int(size_mb * 1024 * 1024))
            files.append(file)
        
        files_with_sizes, stats = analyze_file_sizes(files)
        
        assert len(files_with_sizes) == 3
        assert stats['count'] == 3
        assert stats['max_mb'] == pytest.approx(30, rel=0.1)
        assert stats['min_mb'] == pytest.approx(10, rel=0.1)
        assert stats['avg_mb'] == pytest.approx(20, rel=0.1)
    
    def test_analyze_empty_list(self):
        """Test analysis with empty file list."""
        files_with_sizes, stats = analyze_file_sizes([])
        
        assert len(files_with_sizes) == 0
        assert stats['count'] == 0
        assert stats['max_mb'] == 0
    
    def test_sort_files_by_size(self, tmp_path):
        """Test sorting files by size."""
        # Create test files of different sizes
        files = []
        for i, size_mb in enumerate([30, 10, 20]):
            file = tmp_path / f"test_{i}.laz"
            file.write_bytes(b'x' * int(size_mb * 1024 * 1024))
            files.append((file, file.stat().st_size))
        
        # Sort ascending (smallest first)
        sorted_asc = sort_files_by_size(files, reverse=False)
        assert sorted_asc[0].name == 'test_1.laz'  # 10MB
        assert sorted_asc[-1].name == 'test_0.laz'  # 30MB
        
        # Sort descending (largest first)
        sorted_desc = sort_files_by_size(files, reverse=True)
        assert sorted_desc[0].name == 'test_0.laz'  # 30MB
        assert sorted_desc[-1].name == 'test_1.laz'  # 10MB


class TestLogging:
    """Tests for logging functions."""
    
    def test_log_memory_configuration(self, caplog):
        """Test memory configuration logging."""
        import logging
        
        # Set logging level to capture INFO messages
        caplog.set_level(logging.INFO)
        
        worker_info = {
            'max_file_size_mb': 250,
            'avg_file_size_mb': 150,
            'available_memory_gb': 16.0,
            'memory_per_worker_gb': 3.5,
            'has_memory_pressure': False,
            'recommendation_reason': 'Test reason'
        }
        
        log_memory_configuration(
            num_files=10,
            num_workers=4,
            worker_info=worker_info,
            mode='core'
        )
        
        # Check that key information is logged
        log_text = caplog.text
        assert '10' in log_text  # num_files
        assert '4' in log_text   # num_workers
        assert 'CORE' in log_text.upper()
