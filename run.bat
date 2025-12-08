@REM run.bat - Setup venv, install requirements, dan run aplikasi

@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM 1. Check dan create virtual environment jika belum ada
REM ============================================================

if not exist "venv\" (
    echo [*] Membuat virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment berhasil dibuat
) else (
    echo [OK] Virtual environment sudah ada
)

REM ============================================================
REM 2. Activate virtual environment
REM ============================================================

echo [*] Mengaktifkan virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Gagal mengaktifkan virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment aktif

REM ============================================================
REM 3. Install/upgrade pip dan install requirements
REM ============================================================

if exist "requirements.txt" (
    echo [*] Installing requirements dari requirements.txt...
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Gagal install requirements
        pause
        exit /b 1
    )
    echo [OK] Requirements berhasil diinstall
) else (
    echo [WARNING] requirements.txt tidak ditemukan, skip install
    echo [*] Installing dependencies secara manual...
    pip install -q --upgrade pip
    pip install -q numpy pillow pandas openpyxl scikit-image scipy
    echo [OK] Dependencies berhasil diinstall
)

REM ============================================================
REM 4. Jalankan aplikasi
REM ============================================================

echo.
echo [*] Menjalankan watermark_program_v2.py...
echo ============================================================
python watermark_program_v2.py

REM ============================================================
REM 5. Deactivate venv dan pause
REM ============================================================

echo.
deactivate
pause
