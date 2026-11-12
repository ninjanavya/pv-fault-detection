@echo off
title Solar PV Fault Detection - Streamlit
cd /d "%~dp0"
set KERAS_HOME=%~dp0.keras
set MPLCONFIGDIR=%~dp0.cache\matplotlib
set TF_ENABLE_ONEDNN_OPTS=0
py -3.14 -m streamlit run app.py
pause
