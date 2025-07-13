#!/bin/bash

# Configuration
CONFIG_NAME="${1:-adapter_l2}"  # Default to adapter_l2 if not provided
CHECKPOINT_DIR="exp/${CONFIG_NAME}"
LOG_FILE="${CHECKPOINT_DIR}/training.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Auto-resume training script for config: ${CONFIG_NAME}${NC}"
echo -e "${GREEN}Checkpoint directory: ${CHECKPOINT_DIR}${NC}"

# Create checkpoint directory if it doesn't exist
mkdir -p "${CHECKPOINT_DIR}"

# Function to find the latest checkpoint
find_latest_checkpoint() {
    local latest_checkpoint=""
    local latest_step=0
    
    # Find all checkpoint files and extract the one with highest step number
    for checkpoint in "${CHECKPOINT_DIR}"/checkpoint_*.pt; do
        if [ -f "$checkpoint" ]; then
            # Extract step number from filename (e.g., checkpoint_5k.pt -> 5)
            step=$(basename "$checkpoint" | sed -n 's/checkpoint_\([0-9]*\)k\.pt/\1/p')
            if [ -n "$step" ] && [ "$step" -gt "$latest_step" ]; then
                latest_step=$step
                latest_checkpoint=$checkpoint
            fi
        fi
    done
    
    echo "$latest_checkpoint"
}

# Function to run training
run_training() {
    local checkpoint_arg=""
    local latest_checkpoint=$(find_latest_checkpoint)
    
    if [ -n "$latest_checkpoint" ]; then
        echo -e "${YELLOW}Found checkpoint: ${latest_checkpoint}${NC}"
        checkpoint_arg="checkpoint.resume_from=\"${latest_checkpoint}\""
    else
        echo -e "${YELLOW}No checkpoint found, starting from scratch${NC}"
    fi
    
    # Construct and run the training command
    local cmd="uv run cmds/train_adapter.py ${checkpoint_arg} --config-name ${CONFIG_NAME}"
    echo -e "${GREEN}Running: ${cmd}${NC}"
    
    # Execute the command and capture exit status
    eval $cmd 2>&1 | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

# Main loop with retry logic
MAX_RETRIES=100  # Prevent infinite loops
retry_count=0
wait_time=30  # Wait time in seconds before retry

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo -e "\n${GREEN}[Attempt $((retry_count + 1))/${MAX_RETRIES}] Starting training...${NC}"
    
    # Run training
    run_training
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}Training completed successfully!${NC}"
        break
    else
        echo -e "\n${RED}Training crashed with exit code: ${exit_code}${NC}"
        echo -e "${YELLOW}Waiting ${wait_time} seconds before retry...${NC}"
        
        # Log crash information
        echo "[$(date)] Training crashed, exit code: ${exit_code}" >> "$LOG_FILE"
        
        # Wait before retry
        sleep $wait_time
        
        # Increment retry counter
        ((retry_count++))
        
        # Check if we've reached max retries
        if [ $retry_count -ge $MAX_RETRIES ]; then
            echo -e "${RED}Maximum retries (${MAX_RETRIES}) reached. Exiting.${NC}"
            exit 1
        fi
    fi
done

echo -e "${GREEN}Script finished.${NC}"