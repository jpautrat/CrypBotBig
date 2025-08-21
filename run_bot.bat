@echo off
echo Kraken Trading Bot Launcher
echo ===========================
echo.

:menu
echo Choose an option:
echo 1. Run in simulation mode (no real trades)
echo 2. Run in live trading mode (REAL MONEY)
echo 3. Run with custom settings
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto simulation
if "%choice%"=="2" goto live
if "%choice%"=="3" goto custom
if "%choice%"=="4" goto end

echo Invalid choice. Please try again.
goto menu

:simulation
echo.
echo Starting bot in SIMULATION mode...
python bot.py
goto end

:live
echo.
echo WARNING: You are about to start the bot in LIVE trading mode.
echo This will use REAL MONEY to execute REAL trades.
echo.
set /p confirm="Are you sure you want to continue? (y/n): "
if /i "%confirm%"=="y" (
    echo Starting bot in LIVE trading mode...
    python bot.py --live
) else (
    echo Live trading cancelled.
    goto menu
)
goto end

:custom
echo.
echo Custom Settings
echo ==============
echo.
echo Trading Mode:
echo 1. Simulation mode (no real trades)
echo 2. Live trading mode (REAL MONEY)
set /p mode="Enter mode (1-2): "

echo.
echo Scanning Frequency:
echo 1. Ultra-fast (20 times per second)
echo 2. Fast (10 times per second)
echo 3. Medium (5 times per second)
echo 4. Slow (1 time per second)
echo 5. Custom frequency
set /p freq="Enter frequency (1-5): "

set interval=0.05
if "%freq%"=="1" set interval=0.05
if "%freq%"=="2" set interval=0.1
if "%freq%"=="3" set interval=0.2
if "%freq%"=="4" set interval=1
if "%freq%"=="5" (
    set /p interval="Enter custom interval in seconds (e.g., 0.5): "
)

echo.
echo Profit Threshold:
echo 1. Aggressive (0.1%% after fees)
echo 2. Balanced (0.2%% after fees)
echo 3. Conservative (0.5%% after fees)
echo 4. Custom threshold
set /p profit="Enter profit threshold (1-4): "

set threshold=0.001
if "%profit%"=="1" set threshold=0.001
if "%profit%"=="2" set threshold=0.002
if "%profit%"=="3" set threshold=0.005
if "%profit%"=="4" (
    set /p threshold="Enter custom profit threshold (e.g., 0.003 for 0.3%%): "
)

echo.
echo Initial Buy Strategy:
echo 1. Buy immediately regardless of market conditions
echo 2. Buy only if price is unusually low in 24h range
echo 3. Wait for a dip (don't buy immediately)
set /p buynow="Enter option (1-3): "

set buynowflag=
if "%buynow%"=="1" set buynowflag=--buy-now
if "%buynow%"=="2" set buynowflag=

echo.
echo Starting bot with custom settings...
echo Interval: %interval% seconds
echo Profit threshold: %threshold%
echo Buy immediately: %buynow%

if "%mode%"=="1" (
    python bot.py --interval %interval% --profit %threshold% %buynowflag%
) else (
    echo WARNING: You are about to start the bot in LIVE trading mode.
    echo This will use REAL MONEY to execute REAL trades.
    echo.
    set /p confirm="Are you sure you want to continue? (y/n): "
    if /i "%confirm%"=="y" (
        python bot.py --live --interval %interval% --profit %threshold% %buynowflag%
    ) else (
        echo Live trading cancelled.
        goto menu
    )
)

:end
echo.
echo Bot has been stopped. Press any key to exit...
pause > nul
