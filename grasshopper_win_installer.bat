@echo off
title Raystrack Windows Installer
setlocal EnableExtensions EnableDelayedExpansion

REM ===============================================
REM Raystrack installer for Windows (Rhino 8 only)
REM - Installs the package into Rhino 8 Python (RhinoCode env).
REM - Copies all *.ghuser from rhino\components (all subfolders)
REM   into %APPDATA%\Grasshopper\UserObjects\raystrack (flattened).
REM ===============================================

REM ----------------------------
REM Resolve repo root directory
REM ----------------------------
set "LOG_PREFIX=[raystrack]"
set "REPO_DIR=%~dp0"
for %%# in ("%REPO_DIR%") do set "REPO_DIR=%%~f#"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

REM Optional override via first argument
if not "%~1"=="" (
  if exist "%~1\" (
    set "REPO_DIR=%~1"
    if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
  )
)

echo %LOG_PREFIX% Starting installation...
echo %LOG_PREFIX% Repo root: "%REPO_DIR%"
echo %LOG_PREFIX% A console will stay open to show progress.
echo.

REM ----------------------------------------------
REM 1) Install into Rhino 8 Python environment
REM ----------------------------------------------
echo %LOG_PREFIX% Installing into Rhino 8 Python environment...

set "RHINO_ROOT=%USERPROFILE%\.rhinocode"
set "RHINO_PY_EXE="

REM Prefer the known 3.9 env
if exist "%RHINO_ROOT%\py39-rh8\python.exe" (
  set "RHINO_PY_EXE=%RHINO_ROOT%\py39-rh8\python.exe"
)

REM Fallback: pick the first match from a reverse-sorted list (likely newest)
if not defined RHINO_PY_EXE (
  for /f "delims=" %%i in ('
    dir /b /s "%RHINO_ROOT%\py*-rh8\python.exe" 2^>nul ^| sort /r
  ') do (
    if not defined RHINO_PY_EXE set "RHINO_PY_EXE=%%i"
  )
)

if defined RHINO_PY_EXE (
  echo %LOG_PREFIX% Using Rhino Python at: "%RHINO_PY_EXE%"
  "%RHINO_PY_EXE%" -m pip install --upgrade "%REPO_DIR%"
  if errorlevel 1 (
    echo %LOG_PREFIX% ERROR: pip install failed via Rhino Python interpreter.
    echo %LOG_PREFIX% You can install inside Rhino 8 Python using:
    echo     import sys, subprocess ^&^& subprocess.check_check_call([sys.executable,'-m','pip','install','--upgrade',r'%REPO_DIR%'])
  ) else (
    echo %LOG_PREFIX% Package install completed.
  )
) else (
  echo %LOG_PREFIX% INFO: No Rhino 8 Python found under "%RHINO_ROOT%". Skipping package install.
)

echo.

REM ----------------------------------------------------------------
REM 2) Copy *.ghuser to Grasshopper UserObjects\raystrack
REM    - Safe for spaces; builds a list then copies
REM    - Allows override via RAYSTRACK_SRC env var
REM ----------------------------------------------------------------
set "DEFAULT_SRC=%REPO_DIR%\rhino\components"
set "SRC_COMP=%DEFAULT_SRC%"
if defined RAYSTRACK_SRC set "SRC_COMP=%RAYSTRACK_SRC%"

set "GH_USEROBJ=%APPDATA%\Grasshopper\UserObjects\raystrack"

echo %LOG_PREFIX% Source components: "%SRC_COMP%"
echo %LOG_PREFIX% Target UserObjects: "%GH_USEROBJ%"

if not exist "%APPDATA%\Grasshopper\UserObjects" (
  mkdir "%APPDATA%\Grasshopper\UserObjects" >nul 2>&1
)
if not exist "%GH_USEROBJ%" (
  echo %LOG_PREFIX% Creating UserObjects target: "%GH_USEROBJ%"
  mkdir "%GH_USEROBJ%" >nul 2>&1
)

set /a GHUSER_COUNT=0
set /a GHUSER_FAIL=0
set "LISTFILE=%TEMP%\raystrack_ghuser_list.txt"
del /q "%LISTFILE%" >nul 2>&1

if exist "%SRC_COMP%\" (
  echo %LOG_PREFIX% Searching for *.ghuser files (recursively)...
  dir /S /B /A:-D "%SRC_COMP%\*.ghuser" > "%LISTFILE%" 2>nul
) else (
  echo %LOG_PREFIX% WARNING: Source folder not found: "%SRC_COMP%"
  echo %LOG_PREFIX% Fallback: searching whole repo tree...
  dir /S /B /A:-D "%REPO_DIR%\*.ghuser" > "%LISTFILE%" 2>nul
)

set "GHUSER_FOUND=0"
for /f %%# in ('^<"%LISTFILE%" find /c /v ""') do set "GHUSER_FOUND=%%#"

if %GHUSER_FOUND% gtr 0 (
  echo %LOG_PREFIX% Found %GHUSER_FOUND% file(s). Copying...
  for /f "usebackq delims=" %%F in ("%LISTFILE%") do (
    echo %LOG_PREFIX% Copying: "%%~nxF"
    copy /Y "%%~fF" "%GH_USEROBJ%\" >nul
    if errorlevel 1 (
      set /a GHUSER_FAIL+=1
      echo %LOG_PREFIX% WARNING: Failed to copy "%%~fF"
    ) else (
      set /a GHUSER_COUNT+=1
    )
  )
) else (
  echo %LOG_PREFIX% NOTE: No *.ghuser files found under:
  echo               "%SRC_COMP%"
  echo               (and none in fallback "%REPO_DIR%")
)

del /q "%LISTFILE%" >nul 2>&1
echo %LOG_PREFIX% Copy summary: %GHUSER_COUNT% succeeded, %GHUSER_FAIL% failed, %GHUSER_FOUND% discovered.
echo.

echo %LOG_PREFIX% Done. Press any key to close this window.
pause >nul
exit /b 0
