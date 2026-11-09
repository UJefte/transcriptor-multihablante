@echo off
rem Construye dist\Transcriptor\Transcriptor.exe con PyInstaller.
rem   construir_exe.bat            programa + modelo de huella de voz (Whisper se descarga la primera vez)
rem   construir_exe.bat completo   ademas copia los modelos de Whisper: funciona sin internet
rem No copia voces, galeria ni grabaciones: en el .exe cada quien enrola a su equipo desde cero.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set DEST=dist\Transcriptor
if not exist "%PY%" (
    echo No encuentro el entorno .venv en esta carpeta.
    pause
    exit /b 1
)

echo [1/4] Instalando PyInstaller y la interfaz...
"%PY%" -m pip install --quiet --upgrade pyinstaller pyinstaller-hooks-contrib customtkinter pillow einops rotary-embedding-torch
if errorlevel 1 goto error

echo [2/4] Construyendo el .exe. Tarda de 10 a 20 minutos; el detalle queda en construccion.log
"%PY%" -m PyInstaller --noconfirm --clean transcriptor.spec > construccion.log 2>&1
if errorlevel 1 goto error

echo [3/4] Copiando vocabulario y el modelo de huella de voz...
copy /Y vocabulario.txt "%DEST%\" >nul
copy /Y vocabulario_tecnico.txt "%DEST%\" >nul
copy /Y correcciones.txt "%DEST%\" >nul
copy /Y LEEME.md "%DEST%\" >nul
if exist modelos\ecapa\embedding_model.ckpt (
    xcopy /E /I /Y /Q modelos\ecapa "%DEST%\modelos\ecapa" >nul
) else (
    echo   AVISO: no esta modelos\ecapa; el .exe lo descargara la primera vez ^(necesita internet^).
)

if exist modelos\mossformer2-ss-16k\mossformer2_ss_16k.pt (
    if not exist "%DEST%\modelos\mossformer2-ss-16k" mkdir "%DEST%\modelos\mossformer2-ss-16k"
    copy /Y modelos\mossformer2-ss-16k\mossformer2_ss_16k.pt "%DEST%\modelos\mossformer2-ss-16k\" >nul
) else (
    echo   AVISO: no esta el separador de voces; corre antes descargar_separadores.py
)

if exist modelos\traductor\model.bin (
    xcopy /E /I /Y /Q modelos\traductor "%DEST%\modelos\traductor" >nul
) else (
    echo   AVISO: no esta modelos\traductor; para traducir el ingles corre antes preparar_traductor.py
)

if /I "%~1"=="completo" (
    echo [4/4] Copiando los modelos de Whisper...
    for /d %%S in ("%USERPROFILE%\.cache\huggingface\hub\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo\snapshots\*") do xcopy /E /I /Y /Q "%%S" "%DEST%\modelos\whisper\large-v3-turbo" >nul
    for /d %%S in ("%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots\*") do xcopy /E /I /Y /Q "%%S" "%DEST%\modelos\whisper\small" >nul
) else (
    echo [4/4] Whisper se descargara la primera vez que se abra el .exe en cada computadora.
)

echo.
echo Listo: %DEST%\Transcriptor.exe
echo Para compartirlo, copia la carpeta %DEST% completa.
pause
exit /b 0

:error
echo.
echo Fallo la construccion. Revisa construccion.log
pause
exit /b 1
