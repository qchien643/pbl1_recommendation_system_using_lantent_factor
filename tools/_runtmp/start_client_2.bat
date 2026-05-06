@echo off
title PBL1 Client UI 2
cd /d "D:\Code\Project\DUT_pl\pbl1_recommendation_system_using_lantent_factor\cli"
ping -n 5 127.0.0.1 >nul
cls
call npm start --silent -- 127.0.0.1 2
echo.
echo Client exited. Press any key to close.
pause >nul
