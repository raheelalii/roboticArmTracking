@echo off
echo Setting up Visual Studio environment...

REM Try different Visual Studio paths
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :build
)

if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    goto :build
)

if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :build
)

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :build
)

echo Visual Studio not found. Trying with Build Tools...
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    goto :build
)

echo ERROR: Visual Studio or Build Tools not found!
echo Please install Visual Studio 2019/2022 or Visual Studio Build Tools
pause
exit /b 1

:build
echo Building test_move.cpp as DLL and EXE...

REM Build as DLL for Python integration (64-bit)
cl.exe /EHsc /std:c++17 /MD /LD ^
    /I"frcobot_cpp_sdk\windows\x64_vs2022\include" ^
    "test_move.cpp" ^
    /link ^
    /LIBPATH:"frcobot_cpp_sdk\windows\x64_vs2022\lib64\Release" ^
    cobotAPI.lib ^
    ws2_32.lib ^
    /OUT:"robot_control.dll"

if %ERRORLEVEL% NEQ 0 (
    echo DLL build failed!
    pause
    exit /b 1
)

echo DLL build successful!

REM Also build as EXE for standalone testing (64-bit)
cl.exe /EHsc /std:c++17 /MD ^
    /I"frcobot_cpp_sdk\windows\x64_vs2022\include" ^
    "test_move.cpp" ^
    /link ^
    /LIBPATH:"frcobot_cpp_sdk\windows\x64_vs2022\lib64\Release" ^
    cobotAPI.lib ^
    ws2_32.lib ^
    /OUT:"test_move.exe"

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Copying DLL...
copy "frcobot_cpp_sdk\windows\x64_vs2022\lib64\Release\cobotAPI.dll" . >nul

if %ERRORLEVEL% NEQ 0 (
    echo Warning: Could not copy DLL
)

echo Build successful!
echo Running test_move.exe...
echo.
test_move.exe

pause