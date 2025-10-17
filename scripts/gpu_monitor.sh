#!/bin/bash
# Monitoring GPU simple pour IGN LiDAR HD
# Usage: ./scripts/gpu_monitor.sh [duration_seconds]

DURATION=${1:-300}  # 5 minutes par défaut
LOG_FILE="/tmp/gpu_monitor_$(date +%s).log"

echo "🖥️  GPU Monitoring Started - Duration: ${DURATION}s"
echo "📊 Log file: $LOG_FILE"
echo "📈 Press Ctrl+C to stop early"
echo ""

# Header
echo "Timestamp,GPU_Util%,GPU_Mem_Used_MB,GPU_Mem_Total_MB,GPU_Temp_C" > "$LOG_FILE"

# Monitoring loop
for i in $(seq 1 $DURATION); do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Récupérer les métriques GPU
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
    
    if [ $? -eq 0 ]; then
        # Nettoyer et formater
        GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
        GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')
        GPU_TEMP=$(echo "$GPU_INFO" | cut -d',' -f4 | tr -d ' ')
        
        # Log CSV
        echo "$TIMESTAMP,$GPU_UTIL,$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_TEMP" >> "$LOG_FILE"
        
        # Affichage temps réel avec couleurs
        if [ "$GPU_UTIL" -gt 80 ]; then
            COLOR="\033[32m"  # Vert
            STATUS="EXCELLENT"
        elif [ "$GPU_UTIL" -gt 50 ]; then
            COLOR="\033[33m"  # Jaune
            STATUS="MODERATE"
        elif [ "$GPU_UTIL" -gt 10 ]; then
            COLOR="\033[31m"  # Rouge
            STATUS="POOR"
        else
            COLOR="\033[91m"  # Rouge brillant
            STATUS="CRITICAL"
        fi
        
        printf "\r${COLOR}GPU: %3d%% | Mem: %5d/%5d MB | Temp: %2d°C | %s\033[0m" \
               "$GPU_UTIL" "$GPU_MEM_USED" "$GPU_MEM_TOTAL" "$GPU_TEMP" "$STATUS"
    else
        printf "\rERROR: Cannot read GPU stats"
    fi
    
    sleep 1
done

echo ""
echo ""
echo "📊 Monitoring completed. Summary:"
echo "================================"

# Calculer les statistiques
tail -n +2 "$LOG_FILE" | awk -F',' '
{
    util_sum += $2
    mem_used_sum += $3
    temp_sum += $5
    count++
    
    if ($2 > util_max) util_max = $2
    if (util_min == 0 || $2 < util_min) util_min = $2
    
    if ($3 > mem_max) mem_max = $3
    if (mem_min == 0 || $3 < mem_min) mem_min = $3
}
END {
    if (count > 0) {
        util_avg = util_sum / count
        mem_avg = mem_used_sum / count
        temp_avg = temp_sum / count
        
        printf "GPU Utilization:  Min=%d%%, Max=%d%%, Avg=%.1f%%\n", util_min, util_max, util_avg
        printf "Memory Usage:     Min=%dMB, Max=%dMB, Avg=%.0fMB\n", mem_min, mem_max, mem_avg
        printf "Temperature:      Avg=%.1f°C\n", temp_avg
        
        if (util_avg > 80) {
            print "✅ Performance: EXCELLENT (High GPU utilization)"
        } else if (util_avg > 50) {
            print "⚠️  Performance: MODERATE (Medium GPU utilization)"  
        } else if (util_avg > 10) {
            print "❌ Performance: POOR (Low GPU utilization)"
        } else {
            print "❌ Performance: CRITICAL (Very low GPU utilization)"
        }
    }
}'

echo ""
echo "📁 Detailed log: $LOG_FILE"
echo "💡 Tip: Use this during processing to monitor GPU performance"