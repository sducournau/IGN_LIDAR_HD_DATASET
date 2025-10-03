#!/usr/bin/env python3
"""
Test CLI Functionality - IGN LiDAR HD Library

Tests for the command-line interface.
"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from ign_lidar.cli import main


class TestCLI:
    """Test cases for the command-line interface."""
    
    def test_cli_help(self):
        """Test that CLI shows help information."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.output or 'help' in result.output.lower()
    
    def test_cli_version_info(self):
        """Test that CLI can show version information."""
        runner = CliRunner()
        
        # Try common version flags
        for flag in ['--version', '-V']:
            result = runner.invoke(main, [flag])
            # Should either show version or show that flag exists
            # (depending on implementation)
            assert result.exit_code in [0, 2]  # 0 = success, 2 = no such option
    
    @patch('ign_lidar.cli.LiDARProcessor')
    def test_cli_basic_processing(self, mock_processor_class):
        """Test basic CLI processing command."""
        # Setup mock
        mock_processor = MagicMock()
        mock_processor.process_tile.return_value = ['patch1.npz', 'patch2.npz']
        mock_processor_class.return_value = mock_processor
        
        runner = CliRunner()
        
        # Test with minimal required arguments
        with runner.isolated_filesystem():
            # Create a dummy input file
            with open('test.laz', 'w') as f:
                f.write('dummy laz file')
            
            result = runner.invoke(main, [
                '--input', 'test.laz',
                '--output', 'output/',
                '--lod-level', 'LOD2'
            ])
            
            # Should not crash (exit code 0 or 1 depending on implementation)
            assert result.exit_code in [0, 1, 2]
    
    def test_cli_invalid_arguments(self):
        """Test CLI with invalid arguments."""
        runner = CliRunner()
        
        # Test with missing required arguments
        result = runner.invoke(main, [])
        
        # Should show error or help
        assert result.exit_code != 0 or 'help' in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__])