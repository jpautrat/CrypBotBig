@echo off
echo Starting Elite Trading System...

REM Check if Docker is running
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b
)

REM Create data and logs directories if they don't exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs

REM Build and start containers
docker-compose up --build -d

REM Wait for services to be ready
echo Waiting for services to initialize...
timeout /t 10 /nobreak

REM Open web interface
start http://localhost:8000

echo System started successfully! The web interface is available at http://localhost:8000
pause