@echo off
title Raystrack Windows Installer (Rhino 8)
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

REM ============================================================
REM Raystrack installer for Windows (Rhino 8)
REM - Force-(re)installs the package into Rhino 8 Python.
REM - Copies all *.ghuser from rhino\components (all subfolders).
REM - Deletes any existing %APPDATA%\Grasshopper\UserObjects\raystrack
REM ============================================================

chcp 65001 >nul
set "LOG=[raystrack]"

echo %LOG% Starting installation...
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
echo %LOG% Repo directory: "%REPO_DIR%"

if not exist "%REPO_DIR%\pyproject.toml" (
  echo %LOG% ERROR: pyproject.toml not found in "%REPO_DIR%".
  goto :END_WITH_ERROR
)

REM -- Locate Rhino 8 Python (allow override via env var)
set "RHINO_PY="
if defined RAYSTRACK_RHINO_PY (
  set "RHINO_PY=%RAYSTRACK_RHINO_PY%"
  echo %LOG% Using Rhino Python from RAYSTRACK_RHINO_PY: "%RHINO_PY%"
) else (
  set "RHINO_PY=%USERPROFILE%\.rhinocode\py39-rh8\python.exe"
  if not exist "%RHINO_PY%" (
    echo %LOG% Default py39-rh8 not found, searching for *rh8\python.exe...
    for /f "delims=" %%P in ('dir /b /s "%USERPROFILE%\.rhinocode\py*-rh8\python.exe" 2^>nul') do (
      set "RHINO_PY=%%P"
      goto :FOUND_RHINO_PY
    )
  ) else (
    goto :FOUND_RHINO_PY
  )
)

:FOUND_RHINO_PY
if not exist "%RHINO_PY%" (
  echo %LOG% ERROR: Could not find Rhino 8 Python under "%USERPROFILE%\.rhinocode".
  echo %LOG% Tip: set RAYSTRACK_RHINO_PY to your Rhino 8 python.exe and re-run.
  goto :END_WITH_ERROR
)
echo %LOG% Rhino Python: "%RHINO_PY%"

REM -- Ensure pip is present, then upgrade tooling (best effort)
echo %LOG% Checking pip...
"%RHINO_PY%" -m pip --version >nul 2>&1 || (
  echo %LOG% Bootstrapping pip with ensurepip...
  "%RHINO_PY%" -m ensurepip --upgrade || (
    echo %LOG% ERROR: ensurepip failed. Cannot proceed.
    goto :END_WITH_ERROR
  )
)

echo %LOG% Upgrading pip/setuptools/wheel/build (best effort)...
"%RHINO_PY%" -m pip install --upgrade pip setuptools wheel build >nul 2>&1

REM -- Force-(re)install raystrack from this repo (overwrites existing)
echo %LOG% Installing/overwriting raystrack in Rhino 8 Python...
pushd "%REPO_DIR%" >nul
"%RHINO_PY%" -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps "%REPO_DIR%"
set "INSTALL_RC=%ERRORLEVEL%"
popd >nul
if not "%INSTALL_RC%"=="0" (
  echo %LOG% ERROR: pip install failed with code %INSTALL_RC%.
  goto :END_WITH_ERROR
)
echo %LOG% Package installed (force-reinstalled) successfully.

REM -- Prepare Grasshopper UserObjects destination (delete if exists)
set "GH_USEROBJ_BASE=%APPDATA%\Grasshopper\UserObjects"
set "DST_DIR=%GH_USEROBJ_BASE%\raystrack"

if not exist "%APPDATA%\Grasshopper" mkdir "%APPDATA%\Grasshopper" 2>nul
if not exist "%GH_USEROBJ_BASE%" mkdir "%GH_USEROBJ_BASE%" 2>nul

if exist "%DST_DIR%" (
  echo %LOG% Removing existing destination: "%DST_DIR%"
  rmdir /S /Q "%DST_DIR%"
  if exist "%DST_DIR%" (
    echo %LOG% ERROR: Could not remove "%DST_DIR%".
    goto :END_WITH_ERROR
  )
)

echo %LOG% Creating destination: "%DST_DIR%"
mkdir "%DST_DIR%" 2>nul || (
  echo %LOG% ERROR: Could not create "%DST_DIR%".
  goto :END_WITH_ERROR
)

REM -- Determine search root for .ghuser files
set "PRIMARY_SRC=%REPO_DIR%\rhino\components"
set "SEARCH_ROOT="
if exist "%PRIMARY_SRC%" (
  set "SEARCH_ROOT=%PRIMARY_SRC%"
) else if exist "%REPO_DIR%\components" (
  set "SEARCH_ROOT=%REPO_DIR%\components"
) else (
  set "SEARCH_ROOT=%REPO_DIR%"
)

echo %LOG% Searching for .ghuser files under: "%SEARCH_ROOT%"

set /a FOUND=0
set /a COPIED=0
for /r "%SEARCH_ROOT%" %%F in (*.ghuser) do (
  set /a FOUND+=1
  echo %LOG% Copying "%%F"
  copy /Y "%%F" "%DST_DIR%" >nul && ( set /a COPIED+=1 ) || (
    echo %LOG% WARNING: Failed to copy: %%F
  )
)

if "!FOUND!"=="0" (
  echo %LOG% NOTE: No .ghuser files found under "%SEARCH_ROOT%".
) else (
  echo %LOG% Found !FOUND! .ghuser file^(s^); copied !COPIED! to "%DST_DIR%".
)

goto :END_OK


:END_WITH_ERROR
echo.
echo %LOG% ==================== FAILED ====================
echo %LOG% See messages above. Fix the issue and re-run.
echo %LOG% Press any key to close.
pause >nul
exit /b 1

:END_OK
echo.
echo %LOG% ==================== DONE ====================
echo %LOG% Installation and UserObjects copy steps completed.
echo %LOG% Press any key to close.
pause >nul
exit /b 0
