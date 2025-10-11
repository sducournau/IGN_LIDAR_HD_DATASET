#!/bin/bash
# Monitor GPU utilization during processing
# Shows that vectorized optimization is working

echo "Monitoring GPU during processing..."
echo "=================================="
echo ""

for i in {1..20}; do
    timestamp=$(date "+%H:%M:%S")
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        IFS=',' read -r util mem_used mem_total temp <<< "$gpu_info"
        echo "[$timestamp] GPU: ${util}% | VRAM: ${mem_used}MB/${mem_total}MB | Temp: ${temp}°C"
    else
        echo "[$timestamp] GPU monitoring not available"
    fi
    
    sleep 3
done

echo ""
echo "=================================="
echo "If GPU util is 90-100%, vectorized optimization is working! ✓"
