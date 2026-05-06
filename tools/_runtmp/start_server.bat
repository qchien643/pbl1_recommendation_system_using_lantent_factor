@echo off
title PBL1 Server Dashboard
cd /d "D:\Code\Project\DUT_pl\pbl1_recommendation_system_using_lantent_factor\cli"
cls
call npm run server --silent
echo.
echo Server exited. Press any key to close.
pause >nul
