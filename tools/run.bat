@echo off
REM run.bat — One-shot bootstrap + launch React Ink UI (Windows cmd native).
REM
REM Tu dong:
REM   1. Build C++ binaries (cmake) neu chua co build/server.exe
REM   2. Seed du lieu mau (.tbl tables) neu data/users.tbl chua ton tai
REM   3. Cai cli/node_modules neu chua
REM   4. Mo Server Dashboard + N Client UI moi cai 1 cua so (wt.exe hoac cmd start)
REM
REM Usage:
REM   tools\run.bat            (3 clients - default)
REM   tools\run.bat 5          (5 clients)
REM
REM Yeu cau: Windows + cmake + node 18+ + g++ (MinGW) hoac MSVC trong PATH.

setlocal EnableDelayedExpansion

REM -------- Move to project root --------
cd /d "%~dp0\.."
set "ROOT=%CD%"

set "N=%~1"
if "%N%"=="" set "N=3"

echo ============================================================
echo   PBL1 Restaurant -- Auto launcher (Server + %N% Clients)
echo ============================================================

REM -------- Sanity checks --------
where cmake >nul 2>&1 || (echo [ERROR] Thieu cmake trong PATH & exit /b 1)
where node  >nul 2>&1 || (echo [ERROR] Thieu node ^(>=18^) trong PATH & exit /b 1)
where npm   >nul 2>&1 || (echo [ERROR] Thieu npm trong PATH & exit /b 1)
where g++   >nul 2>&1
if errorlevel 1 (
  where cl >nul 2>&1 || (echo [ERROR] Thieu C++ compiler ^(g++ hoac cl^) trong PATH & exit /b 1)
)

REM -------- 1/4: Build --------
echo.
echo [1/4] Build C++ binaries
if not exist "build\server.exe" (
  if not exist "build" (
    where mingw32-make >nul 2>&1
    if not errorlevel 1 (
      cmake -S . -B build -G "MinGW Makefiles" || exit /b 1
    ) else (
      cmake -S . -B build || exit /b 1
    )
  )
  cmake --build build || exit /b 1
) else (
  echo   build/server.exe da co - bo qua build
)
for %%F in (server.exe client.exe seed_data.exe) do (
  if not exist "build\%%F" (
    echo [ERROR] build\%%F khong ton tai sau khi build & exit /b 1
  )
)
echo   server.exe, client.exe, seed_data.exe san sang

REM -------- 2/4: Seed --------
echo.
echo [2/4] Seed du lieu mau
if exist "data\users.tbl" if exist "data\transactions.tbl" if exist "data\lfm_p.tbl" (
  echo   data\*.tbl da co - bo qua seed
  goto :seed_done
)
echo   Sinh data: 10 personas + ~180 transactions + train LFM ...
build\seed_data.exe >nul || (echo [ERROR] seed_data that bai & exit /b 1)
echo   Seed xong
:seed_done

REM -------- 3/4: npm install --------
echo.
echo [3/4] Cai npm deps cho React Ink UI
if exist "cli\node_modules\ink\package.json" (
  echo   cli\node_modules da co - bo qua npm install
) else (
  echo   cd cli ^&^& npm install ^(lan dau, ~30s^) ...
  pushd cli
  call npm install --silent || (popd & echo [ERROR] npm install that bai & exit /b 1)
  popd
  echo   npm install xong
)

REM -------- 4/4: Launch --------
echo.
echo [4/4] Spawn Server Dashboard + %N% Client UI

REM Sinh wrapper .bat de cmd /k khong bi sai escape
set "TMPDIR=%ROOT%\tools\_runtmp"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"
del /q "%TMPDIR%\*.bat" 2>nul

set "SERVER_BAT=%TMPDIR%\start_server.bat"
(
  echo @echo off
  echo title PBL1 Server Dashboard
  echo cd /d "%ROOT%\cli"
  echo cls
  echo call npm run server --silent
  echo echo.
  echo echo Server exited. Press any key to close.
  echo pause ^>nul
) > "%SERVER_BAT%"

for /L %%i in (1,1,%N%) do (
  set "CBAT=%TMPDIR%\start_client_%%i.bat"
  (
    echo @echo off
    echo title PBL1 Client UI %%i
    echo cd /d "%ROOT%\cli"
    echo ping -n 5 127.0.0.1 ^>nul
    echo cls
    echo call npm start --silent -- 127.0.0.1 %%i
    echo echo.
    echo echo Client exited. Press any key to close.
    echo pause ^>nul
  ) > "!CBAT!"
)

REM Uu tien Windows Terminal (gop tabs); fallback cmd start (mo cua so rieng)
where wt.exe >nul 2>&1
if not errorlevel 1 (
  echo   Dung Windows Terminal ^(wt.exe^) -- gop tabs
  start "" wt.exe -w 0 new-tab --title "SERVER" "%SERVER_BAT%"
  ping -n 5 127.0.0.1 >nul
  for /L %%i in (1,1,%N%) do (
    start "" wt.exe -w 0 new-tab --title "CLIENT-%%i" "%TMPDIR%\start_client_%%i.bat"
    ping -n 2 127.0.0.1 >nul
  )
) else (
  echo   Dung cmd start -- moi UI 1 cua so rieng
  start "SERVER UI" "%SERVER_BAT%"
  ping -n 5 127.0.0.1 >nul
  for /L %%i in (1,1,%N%) do (
    start "CLIENT UI %%i" "%TMPDIR%\start_client_%%i.bat"
    ping -n 2 127.0.0.1 >nul
  )
)

echo.
echo ============================================================
echo   Da mo: 1 Server Dashboard + %N% Client UI
echo ============================================================
echo.
echo   Huong dan:
echo     - Server Dashboard: go 1234 + Enter de MO CA
echo     - Moi Client UI:   nhap SDT 10 chu so, vd:
echo         0901234567 = Anh Nam
echo         0923456789 = Bac Hung
echo         0956789012 = Co Tu
echo     - Dong ca: nhap lai 1234 o Server Dashboard
echo.
echo   Reset fresh: tools\reset.bat
echo ============================================================
endlocal
