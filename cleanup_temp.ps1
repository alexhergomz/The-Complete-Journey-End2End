# Cleanup script - Run this after confirming all files are properly migrated
# This script will remove the old directories and files that have been reorganized

# List of directories to remove after migration is confirmed
$dirsToRemove = @(
    "code",
    "exploration_results",
    "segmentation_results",
    "optimization_results",
    "variable_optimization_results",
    "simulation_results",
    "sirs_model_results",
    "analysis_results",
    "real_transition_models",
    "archive",
    "__pycache__"
)

# List of files to remove after migration is confirmed
$filesToRemove = @(
    # Database and SQL files (moved to data/ or src/database/)
    "dunnhumby.db",
    "create_schema.sql",
    "import_data.sql",
    "setup_postgresql.ps1",
    
    # Images (moved to docs/images/ or results/)
    "revenue_comparison.png",
    "distribution_comparison.png",
    "strategy_comparison.png",
    "simulation_results.png",
    
    # Documentation (moved to docs/guides/)
    "SIRS_model_README.md",
    "optimization_README.md",
    
    # Notebooks (moved to src/notebooks/)
    "initial_exploration.ipynb",
    "retail_analytics_sqlite.ipynb",
    "retail_analytics_exploration.ipynb"
)

Write-Host "This script will remove the following directories:" -ForegroundColor Yellow
foreach ($dir in $dirsToRemove) {
    Write-Host "  - $dir" -ForegroundColor Cyan
}

Write-Host "And the following files from the root directory:" -ForegroundColor Yellow
foreach ($file in $filesToRemove) {
    Write-Host "  - $file" -ForegroundColor Cyan
}

$confirmation = Read-Host "Are you sure you want to proceed? (yes/no)"
if ($confirmation -eq "yes") {
    # Remove directories
    foreach ($dir in $dirsToRemove) {
        if (Test-Path $dir) {
            Write-Host "Removing $dir..." -ForegroundColor Green
            Remove-Item -Path $dir -Recurse -Force
        } else {
            Write-Host "Directory $dir not found, skipping..." -ForegroundColor Gray
        }
    }
    
    # Remove files
    foreach ($file in $filesToRemove) {
        if (Test-Path $file) {
            Write-Host "Removing $file..." -ForegroundColor Green
            Remove-Item -Path $file -Force
        } else {
            Write-Host "File $file not found, skipping..." -ForegroundColor Gray
        }
    }
    
    Write-Host "Cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "Operation cancelled." -ForegroundColor Red
} 