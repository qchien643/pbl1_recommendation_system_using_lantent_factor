@echo off
REM reset.bat — Xoa toan bo data runtime + reseed ve fresh state.
REM
REM Usage:
REM   tools\reset.bat              (xoa + reseed)
REM   tools\reset.bat --no-seed    (chi xoa, khong reseed)

setlocal
cd /d "%~dp0\.."

set "NOSEED=0"
if "%~1"=="--no-seed" set "NOSEED=1"

echo ============================================================
echo   Reset PBL1 data -^> fresh state
echo ============================================================

REM Kill server.exe / client.exe neu dang chay
tasklist /FI "IMAGENAME eq server.exe" 2>nul | find /I "server.exe" >nul
if not errorlevel 1 (
  echo Dung server.exe ...
  taskkill /F /IM server.exe >nul 2>&1
)
tasklist /FI "IMAGENAME eq client.exe" 2>nul | find /I "client.exe" >nul
if not errorlevel 1 (
  taskkill /F /IM client.exe >nul 2>&1
)

REM Xoa data
echo Xoa data files ...
del /q data\*.tbl 2>nul
del /q data\*.dat 2>nul
del /q data\transactions.log 2>nul
del /q data\personas.txt 2>nul
del /q data\reports\*.txt 2>nul
echo   Da xoa: *.tbl, *.dat, transactions.log, personas.txt, reports\*.txt

if "%NOSEED%"=="1" (
  echo.
  echo Reset xong. ^(--no-seed^) -- khong reseed.
  endlocal
  exit /b 0
)

REM Reseed
echo.
echo Reseed du lieu mau ...
if not exist "build\seed_data.exe" (
  echo   build\seed_data.exe chua co - chay cmake build ...
  where mingw32-make >nul 2>&1
  if not errorlevel 1 (
    cmake -S . -B build -G "MinGW Makefiles" >nul || exit /b 1
  ) else (
    cmake -S . -B build >nul || exit /b 1
  )
  cmake --build build --target seed_data || exit /b 1
)
build\seed_data.exe >nul || (echo [ERROR] seed_data that bai & exit /b 1)
echo   Seed xong

echo.
echo ============================================================
echo   Reset + reseed xong. Chay: tools\run.bat
echo ============================================================
endlocal
