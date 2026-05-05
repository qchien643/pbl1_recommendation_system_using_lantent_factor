@echo off
REM launch_multi.bat — Mo 1 server + N client trong cac cua so terminal rieng biet.
REM Su dung cmd /k de giu cua so mo sau khi chuong trinh chay.
REM Chay: tools\launch_multi.bat [so_client]
REM       tools\launch_multi.bat 3      → mo 1 server + 3 client

setlocal
set NCLIENTS=%1
if "%NCLIENTS%"=="" set NCLIENTS=3

REM Chuyen ve thu muc goc du an
cd /d "%~dp0\.."

if not exist "build\server.exe" (
  echo [ERROR] build\server.exe khong ton tai.
  echo Chay: cmake -S . -B build -G "MinGW Makefiles" ^&^& cmake --build build
  pause
  exit /b 1
)

echo [Launcher] Starting 1 server + %NCLIENTS% clients on 127.0.0.1:8888

REM Mo server trong cua so rieng
start "SERVER" cmd /k "cd /d %cd% && echo === SERVER === && build\server.exe --server"

REM Cho server start
timeout /t 2 /nobreak > nul

REM Mo N client, moi cai mot cua so
for /L %%i in (1,1,%NCLIENTS%) do (
  start "CLIENT %%i" cmd /k "cd /d %cd% && echo === CLIENT BAN %%i === && build\client.exe --client 127.0.0.1 %%i"
  timeout /t 1 /nobreak > nul
)

echo [Launcher] Da mo %NCLIENTS% client + 1 server. Sang cua so SERVER va nhap ma so 1234 de mo ca.
echo [Launcher] Dong tat ca cua so bang cach dong tung cai, hoac Ctrl+C trong moi cua so.
endlocal
