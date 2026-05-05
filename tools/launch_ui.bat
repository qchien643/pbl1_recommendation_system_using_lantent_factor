@echo off
REM launch_ui.bat — Mo React Ink Server + N React Ink Client trong cua so rieng.
REM Usage: tools\launch_ui.bat [so_client=3]

setlocal
set NCLIENTS=%1
if "%NCLIENTS%"=="" set NCLIENTS=3

cd /d "%~dp0\.."

if not exist "build\server.exe" (
  echo [ERROR] build\server.exe khong ton tai. Chay cmake --build build truoc.
  pause
  exit /b 1
)
if not exist "cli\node_modules\ink\package.json" (
  echo [ERROR] cli\node_modules chua co. Chay: cd cli ^&^& npm install
  pause
  exit /b 1
)

echo [Launcher] Starting React Ink Server + %NCLIENTS% React Ink Clients

REM Server dashboard
start "SERVER UI" cmd /k "cd /d %cd%\cli && npm run server"

timeout /t 3 /nobreak > nul

for /L %%i in (1,1,%NCLIENTS%) do (
  start "CLIENT UI %%i" cmd /k "cd /d %cd%\cli && npm start -- 127.0.0.1 %%i"
  timeout /t 1 /nobreak > nul
)

echo [Launcher] Da mo SERVER UI + %NCLIENTS% CLIENT UI.
echo [Launcher] Tai SERVER UI: nhap MA SO (vd 1234) + Enter de mo ca.
echo [Launcher] Tai moi CLIENT UI: nhap SDT (thu 0901234567, 0923456789, 0956789012).
endlocal
