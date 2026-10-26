<#
.SYNOPSIS
    PCBDetect Automation Script for Windows PowerShell
.DESCRIPTION
    Helper script to run common tasks such as EDA, testing, linting, formatting.
.EXAMPLE
    .\tasks.ps1 eda
    .\tasks.ps1 test
    .\tasks.ps1 lint
#>

param (
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("prepare-data", "eda", "test", "lint", "format", "benchmark", "help")]
    [string]$Task
)

$Python = ".\.venv\Scripts\python.exe"
$Pytest = ".\.venv\Scripts\pytest.exe"
$Ruff   = ".\.venv\Scripts\ruff.exe"

switch ($Task) {
    "benchmark" {
        Write-Host "Running Milestone 2 Comparative Benchmark (Classical vs YOLO)..." -ForegroundColor Cyan
        & $Python -m pcbdetect.eval.benchmark_m2
    }
    "prepare-data" {
        Write-Host "Running Data Pipeline: Exporting to YOLO and COCO (Strict Group-Disjoint)..." -ForegroundColor Cyan
        & $Python -m pcbdetect.data.convert --strategy group_disjoint --format both
    }
    "eda" {
        Write-Host "Running Exploratory Data Analysis..." -ForegroundColor Cyan
        & $Python -m pcbdetect.data.eda
    }
    "test" {
        Write-Host "Running test suite..." -ForegroundColor Cyan
        & $Pytest -v tests/
    }
    "lint" {
        Write-Host "Running Ruff linter..." -ForegroundColor Cyan
        & $Ruff check .
    }
    "format" {
        Write-Host "Formatting code with Ruff..." -ForegroundColor Cyan
        & $Ruff format .
    }
    "help" {
        Write-Host "Available tasks: prepare-data, eda, test, lint, format, benchmark" -ForegroundColor Green
    }
}
