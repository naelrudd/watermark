@echo off
echo ==============================
echo  CEK VERSI PYTHON
echo ==============================
python --version

echo.
echo ==============================
echo  INSTALL DEPENDENCIES
echo ==============================
pip install numpy pillow scikit-image scipy pandas

echo.
echo ==============================
echo  MENJALANKAN PROGRAM
echo ==============================
python watermark_program.py

echo.
echo ==============================
echo  SELESAI
echo  - Statistik ringkasan ada di console
echo  - File hasil: watermark_evaluation_results.csv
echo ==============================

pause
