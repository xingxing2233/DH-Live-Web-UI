@echo off
set "PATH=%CD%;%PATH%"
set PYTHONUSERBASE=kelong\Lib\site-packages
kelong\Scripts\python.exe webapp.py
pause