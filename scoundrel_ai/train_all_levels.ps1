# Train DQN model at each strategy level (BASIC, INTERMEDIATE, ADVANCED, EXPERT)
# Usage: .\train_all_levels.ps1 [timesteps] [algorithm]
# Example: .\train_all_levels.ps1 250000 dqn

param(
    [int]$Timesteps = 100000,
    [string]$Algorithm = "dqn"
)

# Strategy levels
$Levels = @(1, 2, 3, 4)
$LevelNames = @("BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT")

# Write-Host "==================================================" -ForegroundColor Cyan
# Write-Host "Training $($Algorithm.ToUpper()) at all strategy levels" -ForegroundColor Cyan
# Write-Host "Timesteps per level: $Timesteps" -ForegroundColor Cyan
# Write-Host "==================================================" -ForegroundColor Cyan
# Write-Host ""

# Train at each level
for ($i = 0; $i -lt $Levels.Length; $i++) {
    $Level = $Levels[$i]
    $LevelName = $LevelNames[$i]
    
    # Write-Host ""
    # Write-Host "==================================================" -ForegroundColor Yellow
    # Write-Host "Training Level $Level : $LevelName" -ForegroundColor Yellow
    # Write-Host "==================================================" -ForegroundColor Yellow
    # Write-Host ""
    
    python -m scoundrel_ai.train_dqn `
        --mode train `
        --algorithm $Algorithm `
        --timesteps $Timesteps `
        --strategy-level $Level
    
    # Check if training succeeded
    if ($LASTEXITCODE -eq 0) {
        # Write-Host ""
        # Write-Host "✓ Level $Level ($LevelName) training completed successfully" -ForegroundColor Green
        
        # Evaluate the trained model
        # Write-Host ""
        # Write-Host "==================================================" -ForegroundColor Cyan
        # Write-Host "Evaluating Level $Level : $LevelName" -ForegroundColor Cyan
        # Write-Host "==================================================" -ForegroundColor Cyan
        # Write-Host ""
        
        python -m scoundrel_ai.train_dqn `
            --mode eval `
            --model-path "scoundrel_ai/models/scoundrel_$Algorithm/best_model" `
            --episodes 100
        
        if ($LASTEXITCODE -eq 0) {
            # Write-Host ""
            # Write-Host "✓ Level $Level ($LevelName) evaluation completed successfully" -ForegroundColor Green
        } else {
            # Write-Host ""
            # Write-Host "✗ Level $Level ($LevelName) evaluation failed" -ForegroundColor Red
        }
    } else {
        # Write-Host ""
        # Write-Host "✗ Level $Level ($LevelName) training failed" -ForegroundColor Red
        exit 1
    }
}

# Write-Host ""
# Write-Host "==================================================" -ForegroundColor Green
# Write-Host "All levels trained successfully!" -ForegroundColor Green
# Write-Host "==================================================" -ForegroundColor Green
# Write-Host ""
# Write-Host "To view results in TensorBoard:" -ForegroundColor Cyan
# Write-Host "  tensorboard --logdir ./scoundrel_ai/tensorboard_logs/" -ForegroundColor White
# Write-Host ""
