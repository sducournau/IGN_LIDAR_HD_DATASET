"""CLI utility functions for validation and common operations."""

from pathlib import Path
from typing import Optional, Callable, Iterable, TypeVar, Any
from multiprocessing import Pool
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def validate_input_path(
    path: Path,
    must_exist: bool = True,
    path_type: str = "directory"
) -> bool:
    """Validate input path exists and is correct type.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        path_type: Type of path ('file' or 'directory')
    
    Returns:
        True if valid, False otherwise
    """
    if must_exist and not path.exists():
        logger.error(f"Input {path_type} not found: {path}")
        return False
    
    if path.exists():
        if path_type == "file" and not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if path_type == "directory" and not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return False
    
    return True


def ensure_output_dir(output_dir: Path) -> bool:
    """Ensure output directory exists.
    
    Args:
        output_dir: Directory to create if needed
    
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return False


def discover_laz_files(
    input_path: Path,
    recursive: bool = True,
    max_files: Optional[int] = None,
    pattern: str = "*.laz"
) -> list[Path]:
    """Discover LAZ files in path.
    
    Args:
        input_path: Directory or file to search
        recursive: Whether to search recursively
        max_files: Maximum number of files to return
        pattern: File pattern to match (default: '*.laz')
    
    Returns:
        List of LAZ file paths sorted by name
    """
    if input_path.is_file():
        if input_path.suffix.lower() == '.laz':
            return [input_path]
        else:
            logger.warning(f"Input file is not a LAZ file: {input_path}")
            return []
    
    # Search pattern
    search_pattern = f'**/{pattern}' if recursive else pattern
    files = sorted(input_path.glob(search_pattern))
    
    if max_files and len(files) > max_files:
        logger.info(f"Limiting to first {max_files} files (found {len(files)} total)")
        files = files[:max_files]
    
    return files


def process_with_progress(
    items: Iterable[T],
    worker_func: Callable[[T], R],
    description: str = "Processing",
    num_workers: Optional[int] = None,
    disable_parallel: bool = False,
    chunksize: int = 1
) -> list[R]:
    """Process items with progress bar and optional parallelization.
    
    Args:
        items: Items to process
        worker_func: Function to apply to each item
        description: Progress bar description
        num_workers: Number of parallel workers (None = CPU count)
        disable_parallel: Force sequential processing
        chunksize: Number of items per worker chunk
    
    Returns:
        List of results in arbitrary order (due to unordered parallel processing)
    """
    items_list = list(items)
    
    if not items_list:
        logger.warning("No items to process")
        return []
    
    if disable_parallel or len(items_list) == 1:
        # Sequential processing
        logger.info(f"Processing {len(items_list)} item(s) sequentially")
        results = []
        for item in tqdm(items_list, desc=description):
            try:
                result = worker_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {item}: {e}", exc_info=True)
                results.append(None)
        return results
    
    # Parallel processing
    logger.info(f"Processing {len(items_list)} items with {num_workers or 'auto'} workers")
    with Pool(processes=num_workers) as pool:
        results = []
        with tqdm(total=len(items_list), desc=description) as pbar:
            for result in pool.imap_unordered(worker_func, items_list, chunksize=chunksize):
                results.append(result)
                pbar.update()
    
    return results


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB", "230 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def log_processing_summary(
    total_files: int,
    success_count: int,
    failed_files: Optional[list[Path]] = None,
    operation: str = "Processing"
) -> None:
    """Log processing summary with standardized format.
    
    Args:
        total_files: Total number of files processed
        success_count: Number of successful operations
        failed_files: List of files that failed (optional)
        operation: Operation name for logging
    """
    logger.info("=" * 70)
    
    if success_count == total_files:
        logger.info(f"✓ {operation} complete: {success_count}/{total_files} files succeeded")
    else:
        failed_count = total_files - success_count
        logger.warning(
            f"⚠ {operation} completed with issues: "
            f"{success_count} succeeded, {failed_count} failed"
        )
        
        if failed_files:
            logger.warning(f"Failed files:")
            for file_path in failed_files[:10]:  # Limit to first 10
                logger.warning(f"  - {file_path}")
            if len(failed_files) > 10:
                logger.warning(f"  ... and {len(failed_files) - 10} more")
    
    logger.info("=" * 70)


def get_input_output_paths(
    input_arg: Optional[str],
    input_dir_arg: Optional[str],
    output_dir_arg: str
) -> tuple[Optional[Path], Path]:
    """Get and validate input/output paths from arguments.
    
    Args:
        input_arg: Single input file argument
        input_dir_arg: Input directory argument
        output_dir_arg: Output directory argument
    
    Returns:
        Tuple of (input_path, output_dir)
    """
    # Determine input path
    if input_arg:
        input_path = Path(input_arg)
    elif input_dir_arg:
        input_path = Path(input_dir_arg)
    else:
        input_path = None
    
    output_dir = Path(output_dir_arg)
    
    return input_path, output_dir
