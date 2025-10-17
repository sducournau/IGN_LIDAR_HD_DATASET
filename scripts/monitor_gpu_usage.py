#!/usr/bin/env python3
"""
GPU Usage Monitor
Monitor GPU utilization during LiDAR processing
"""

import subprocess
import time
import threading
import signal
import sys
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"🔍 GPU monitoring started (interval: {self.interval}s)")
        print(f"📊 Timestamp, GPU%, Memory%, Temperature(°C), Power(W)")
        print("-" * 60)
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("\n🛑 GPU monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Query GPU stats using nvidia-smi
                cmd = [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    gpu_stats = result.stdout.strip().split(', ')
                    
                    if len(gpu_stats) >= 4:
                        gpu_util = gpu_stats[0]
                        mem_util = gpu_stats[1] 
                        temp = gpu_stats[2]
                        power = gpu_stats[3]
                        
                        print(f"{timestamp}, {gpu_util:>3}%, {mem_util:>3}%, {temp:>3}°C, {power:>6}W")
                    else:
                        print(f"{timestamp}, Error parsing GPU stats")
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp}, nvidia-smi error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"{datetime.now().strftime('%H:%M:%S')}, nvidia-smi timeout")
            except Exception as e:
                print(f"{datetime.now().strftime('%H:%M:%S')}, Monitor error: {e}")
                
            time.sleep(self.interval)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Stopping GPU monitor...")
    sys.exit(0)

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start monitor
    monitor = GPUMonitor(interval=2.0)  # Check every 2 seconds
    monitor.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == "__main__":
    main()