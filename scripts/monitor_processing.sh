#!/bin/bash
# Monitor IGN LiDAR HD processing in real-time
# Usage: ./scripts/monitor_processing.sh

echo "==================================================================="
echo "IGN LiDAR HD Processing Monitor"
echo "==================================================================="
echo ""

while true; do
    clear
    echo "==================================================================="
    echo "System Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==================================================================="
    echo ""
    
    # GPU Status
    if command -v nvidia-smi &> /dev/null; then
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU: %s | Temp: %s°C | Util: %s%% | VRAM: %s/%s MB (%s%% used)\n", $2, $3, $4, $6, $7, $5}'
        echo ""
    fi
    
    # CPU & Memory
    echo "--- System Resources ---"
    echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
    echo "  Memory: $(free -h | awk '/^Mem:/ {printf "%s / %s (%.1f%% used)\n", $3, $2, ($3/$2)*100}')"
    echo ""
    
    # Process Info
    echo "--- IGN LiDAR HD Process ---"
    if pgrep -f "ign-lidar-hd process" > /dev/null; then
        pid=$(pgrep -f "ign-lidar-hd process" | head -n1)
        echo "  PID: $pid"
        echo "  CPU%: $(ps -p $pid -o %cpu --no-headers 2>/dev/null || echo 'N/A')"
        echo "  MEM%: $(ps -p $pid -o %mem --no-headers 2>/dev/null || echo 'N/A')"
        echo "  Runtime: $(ps -p $pid -o etime --no-headers 2>/dev/null || echo 'N/A')"
        echo "  Status: RUNNING ✅"
    else
        echo "  Status: NOT RUNNING ❌"
    fi
    echo ""
    
    # File Progress
    if [ -d "/mnt/c/Users/Simon/ign_lidar/training_simple_50m/enriched_tiles" ]; then
        echo "--- Output Progress ---"
        enriched_count=$(find /mnt/c/Users/Simon/ign_lidar/training_simple_50m/enriched_tiles -name "*.laz" 2>/dev/null | wc -l)
        input_count=$(find /mnt/c/Users/Simon/ign_lidar/unified_dataset_rgb -name "*.laz" 2>/dev/null | wc -l)
        if [ "$input_count" -gt 0 ]; then
            progress=$((enriched_count * 100 / input_count))
            echo "  Processed tiles: $enriched_count / $input_count ($progress%)"
        else
            echo "  Processed tiles: $enriched_count"
        fi
        echo ""
    fi
    
    echo "==================================================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "==================================================================="
    
    sleep 2
done
