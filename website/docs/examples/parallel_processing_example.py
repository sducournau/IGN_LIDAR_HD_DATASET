# Parallel Processing Example

This example demonstrates how to process multiple LiDAR tiles in parallel using the IGN LiDAR HD library.

## Basic Parallel Processing

```python
import multiprocessing as mp
from pathlib import Path
from ign_lidar import Processor, Config

def process_single_tile(args):
    """Process a single tile - function for multiprocessing."""
    input_file, output_dir, config = args
    
    processor = Processor(config)
    output_file = Path(output_dir) / f"{Path(input_file).stem}_processed.las"
    
    try:
        result = processor.process_tile(input_file, str(output_file))
        return {"file": input_file, "status": "success", "result": result}
    except Exception as e:
        return {"file": input_file, "status": "error", "error": str(e)}

def parallel_processing_example():
    """Main parallel processing function."""
    
    # Configuration
    config = Config(
        feature_types=["height_above_ground", "intensity_normalized"],
        classification_model="random_forest",
        chunk_size=1000000
    )
    
    # Input files
    input_directory = Path("data/raw_tiles")
    output_directory = Path("data/processed_tiles")
    output_directory.mkdir(exist_ok=True)
    
    # Find all LAS/LAZ files
    input_files = list(input_directory.glob("*.las")) + list(input_directory.glob("*.laz"))
    
    # Prepare arguments for multiprocessing
    args = [(str(file), str(output_directory), config) for file in input_files]
    
    # Process files in parallel
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_tile, args)
    
    # Process results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed to process: {len(failed)} files")
    
    if failed:
        print("Failed files:")
        for failure in failed:
            print(f"  {failure['file']}: {failure['error']}")
    
    return results

if __name__ == "__main__":
    results = parallel_processing_example()
```

## Advanced Parallel Processing with Progress Tracking

```python
from tqdm import tqdm
import concurrent.futures
from ign_lidar import Processor, Config

class ParallelProcessor:
    def __init__(self, config, max_workers=None):
        self.config = config
        self.max_workers = max_workers or mp.cpu_count() - 1
        
    def process_tile_with_progress(self, input_file, output_dir):
        """Process single tile with detailed progress information."""
        processor = Processor(self.config, verbose=False)
        output_file = Path(output_dir) / f"{Path(input_file).stem}_processed.las"
        
        start_time = time.time()
        
        try:
            result = processor.process_tile(str(input_file), str(output_file))
            processing_time = time.time() - start_time
            
            return {
                "file": input_file,
                "status": "success",
                "processing_time": processing_time,
                "points_processed": result.get("points_count", 0),
                "output_size": output_file.stat().st_size if output_file.exists() else 0
            }
        except Exception as e:
            return {
                "file": input_file,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_files(self, input_files, output_directory):
        """Process multiple files with progress tracking."""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_tile_with_progress, file, output_directory): file 
                for file in input_files
            }
            
            # Process results with progress bar
            results = []
            with tqdm(total=len(input_files), desc="Processing tiles") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar description
                    if result["status"] == "success":
                        pbar.set_postfix({"Current": Path(result["file"]).name[:20] + "..."})
                    else:
                        pbar.set_postfix({"Failed": Path(result["file"]).name[:20] + "..."})
                    
                    pbar.update(1)
            
            return results

# Usage example
def advanced_parallel_example():
    config = Config(
        feature_types=["height_above_ground", "local_density"],
        use_gpu=True,
        memory_limit=8.0
    )
    
    processor = ParallelProcessor(config, max_workers=8)
    
    input_files = Path("data/tiles").glob("*.las")
    output_dir = Path("data/processed")
    
    results = processor.process_files(list(input_files), output_dir)
    
    # Generate summary report
    successful_results = [r for r in results if r["status"] == "success"]
    
    if successful_results:
        total_points = sum(r["points_processed"] for r in successful_results)
        total_time = sum(r["processing_time"] for r in successful_results)
        avg_points_per_sec = total_points / total_time if total_time > 0 else 0
        
        print(f"Processed {total_points:,} points in {total_time:.2f} seconds")
        print(f"Average processing speed: {avg_points_per_sec:,.0f} points/second")
```

## Memory-Efficient Processing for Large Datasets

```python
def memory_efficient_parallel_processing():
    """Process large datasets while managing memory usage."""
    
    config = Config(
        chunk_size=500000,  # Smaller chunks to save memory
        feature_types=["height_above_ground"],
        memory_limit=4.0,   # Limit memory usage
        cache_features=False  # Disable caching to save memory
    )
    
    # Process files in batches to avoid memory overload
    batch_size = 4  # Process 4 files simultaneously
    
    input_files = list(Path("data/large_tiles").glob("*.las"))
    output_dir = Path("data/processed_large")
    
    all_results = []
    
    for i in range(0, len(input_files), batch_size):
        batch = input_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(input_files)-1)//batch_size + 1}")
        
        # Process current batch
        args = [(str(file), str(output_dir), config) for file in batch]
        
        with mp.Pool(processes=min(len(batch), mp.cpu_count() - 1)) as pool:
            batch_results = pool.map(process_single_tile, args)
        
        all_results.extend(batch_results)
        
        # Optional: Force garbage collection between batches
        import gc
        gc.collect()
    
    return all_results
```