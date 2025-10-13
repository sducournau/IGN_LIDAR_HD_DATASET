"""Unified logging configuration for IGN LiDAR HD.

This module provides consistent logging setup across the entire package,
replacing scattered print() statements and inconsistent logging patterns.

Usage:
    # At package initialization
    from ign_lidar.core.logging_config import setup_logging
    setup_logging(level="INFO", log_file="processing.log")
    
    # In any module
    from ign_lidar.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started...")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    include_timestamp: bool = True,
    format_style: str = "detailed"
) -> None:
    """
    Setup consistent logging across the package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        include_timestamp: Include timestamp in log format
        format_style: Formatting style - 'detailed', 'simple', or 'minimal'
    
    Examples:
        # Basic setup
        setup_logging(level="INFO")
        
        # With log file
        setup_logging(level="DEBUG", log_file=Path("logs/processing.log"))
        
        # Minimal format for clean output
        setup_logging(level="INFO", format_style="minimal")
    """
    # Choose format based on style
    if format_style == "detailed":
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            date_format = "%Y-%m-%d %H:%M:%S"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
            date_format = None
    elif format_style == "simple":
        log_format = "[%(levelname)s] %(name)s - %(message)s"
        date_format = None
    elif format_style == "minimal":
        log_format = "%(levelname)s: %(message)s"
        date_format = None
    else:
        raise ValueError(f"Unknown format_style: {format_style}")
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # Create parent directory if needed
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={level}, format={format_style}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Configured logger instance
    
    Examples:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


def enable_debug_for_module(module_name: str) -> None:
    """
    Enable DEBUG level for a specific module without affecting others.
    
    Args:
        module_name: Name of the module (e.g., 'ign_lidar.core.processor')
    
    Examples:
        # Debug only the processor module
        enable_debug_for_module('ign_lidar.core.processor')
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)


def add_feature_flow_logger() -> logging.Logger:
    """
    Create a specialized logger for tracking feature flow through the pipeline.
    
    This is useful for debugging the feature loss bug by tracking how features
    move between different processing stages.
    
    Returns:
        Logger configured for feature flow debugging
    
    Examples:
        feature_logger = add_feature_flow_logger()
        feature_logger.debug(f"Features after computation: {list(features.keys())}")
    """
    logger = logging.getLogger("ign_lidar.features.flow")
    logger.setLevel(logging.DEBUG)
    
    # Add custom handler with specific format
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("🔍 [FEATURE_FLOW] %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False  # Don't propagate to root logger
    
    return logger


class PerformanceLogger:
    """Context manager for logging execution time of code blocks.
    
    Examples:
        with PerformanceLogger("Feature computation"):
            features = compute_features(xyz)
        # Logs: "Feature computation completed in 2.34s"
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize performance logger.
        
        Args:
            operation_name: Name of operation to time
            logger: Logger to use (creates default if None)
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"⏱️  Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        import time
        if self.start_time is None:
            self.logger.error("PerformanceLogger: start_time is None")
            return False
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"⏱️  {self.operation_name} completed in {duration:.2f}s")
        else:
            self.logger.error(
                f"⏱️  {self.operation_name} failed after {duration:.2f}s: {exc_val}"
            )
        return False  # Don't suppress exceptions


# Convenience function for common use case
def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a section header for better readability in logs.
    
    Args:
        title: Section title
        logger: Logger to use (uses root logger if None)
    
    Examples:
        log_section("Processing Tile XYZ")
        # Logs: "========== Processing Tile XYZ =========="
    """
    if logger is None:
        logger = logging.getLogger()
    
    separator = "=" * 60
    logger.info(separator)
    logger.info(f"  {title}")
    logger.info(separator)
