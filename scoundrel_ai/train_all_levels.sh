#!/bin/bash
# Train DQN model at each strategy level (BASIC, INTERMEDIATE, ADVANCED, EXPERT)
# Usage: ./train_all_levels.sh [timesteps] [algorithm]
# Example: ./train_all_levels.sh 250000 dqn

# Default values
TIMESTEPS=${1:-100000}
ALGORITHM=${2:-dqn}

# Strategy levels
LEVELS=("1" "2" "3" "4")
LEVEL_NAMES=("BASIC" "INTERMEDIATE" "ADVANCED" "EXPERT")

echo "=================================================="
echo "Training ${ALGORITHM^^} at all strategy levels"
echo "Timesteps per level: $TIMESTEPS"
echo "=================================================="
echo ""

# Train at each level
for i in "${!LEVELS[@]}"; do
    LEVEL="${LEVELS[$i]}"
    LEVEL_NAME="${LEVEL_NAMES[$i]}"
    
    echo ""
    echo "=================================================="
    echo "Training Level $LEVEL: $LEVEL_NAME"
    echo "=================================================="
    echo ""
    
    python -m scoundrel_ai.train_dqn \
        --mode train \
        --algorithm "$ALGORITHM" \
        --timesteps "$TIMESTEPS" \
        --strategy-level "$LEVEL"
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Level $LEVEL ($LEVEL_NAME) training completed successfully"
        
        # Evaluate the trained model
        echo ""
        echo "=================================================="
        echo "Evaluating Level $LEVEL: $LEVEL_NAME"
        echo "=================================================="
        echo ""
        
        python -m scoundrel_ai.train_dqn \
            --mode eval \
            --model-path "scoundrel_ai/models/scoundrel_${ALGORITHM}/best_model" \
            --episodes 100
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Level $LEVEL ($LEVEL_NAME) evaluation completed successfully"
        else
            echo ""
            echo "✗ Level $LEVEL ($LEVEL_NAME) evaluation failed"
        fi
    else
        echo ""
        echo "✗ Level $LEVEL ($LEVEL_NAME) training failed"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "All levels trained successfully!"
echo "=================================================="
echo ""
echo "To view results in TensorBoard:"
echo "  tensorboard --logdir ./scoundrel_ai/tensorboard_logs/"
echo ""
