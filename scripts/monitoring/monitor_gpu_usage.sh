#!/bin/bash
# Monitor GPU usage in real-time while processing
# Run this in a separate terminal while your ign-lidar-hd process command is running

echo "Monitoring GPU usage - Press Ctrl+C to stop"
echo "=============================================="
echo ""

watch -n 1 'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -F, '\''{printf "Time: %s\nGPU: %s\nGPU Util: %s%%\nMem Util: %s%%\nMemory: %s/%s MB\nTemp: %s°C\nPower: %s W\n", $1, $2, $3, $4, $5, $6, $7, $8}'\'''
