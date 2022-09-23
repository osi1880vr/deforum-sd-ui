@echo off
:: This is heavily stolen from hlky

:: Run all commands using this script's directory as the working directory
cd %~dp0

:: copy over the first line from environment.yaml, e.g. name: ldm, and take the second word after splitting by ":" delimiter
for /F "tokens=2 delims=: " %%i in (environment.yaml) DO (
  set v_conda_env_name=%%i
  goto EOL
)
:EOL

echo Environment name is set as %v_conda_env_name% as per environment.yaml

:: Put the path to conda directory in a file called "custom-conda-path.txt" if it's installed at non-standard path
IF EXIST custom-conda-path.txt (
  FOR /F %%i IN (custom-conda-path.txt) DO set v_custom_path=%%i
)

set v_paths=%ProgramData%\miniconda3
set v_paths=%v_paths%;%USERPROFILE%\miniconda3
set v_paths=%v_paths%;%ProgramData%\anaconda3
set v_paths=%v_paths%;%USERPROFILE%\anaconda3


for %%a in (%v_paths%) do (
  IF NOT "%v_custom_path%"=="" (
    set v_paths=%v_custom_path%;%v_paths%
  )
)

for %%a in (%v_paths%) do (
  if EXIST "%%a\Scripts\activate.bat" (
    SET v_conda_path=%%a
    echo anaconda3/miniconda3 detected in %%a
    goto :CONDA_FOUND
  )
)

IF "%v_conda_path%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  pause
  exit /b 1
)

:CONDA_FOUND

if not exist "z_version_env.tmp" (
  :: first time running, we need to update
  set AUTO=1
  echo we run the first time setup
  call "update_to_latest.cmd"
)

call "%v_conda_path%\Scripts\activate.bat" "%v_conda_env_name%"

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "content\models\sd-v1-4.ckpt" (
  :: python -m streamlit run scripts\webui_streamlit.py --theme.base dark
  echo all ready to fly

) ELSE (
  echo Your model file does not exist! Place it in 'content\models\sd-v1-4.ckpt' with the name 'model.ckpt'.
  pause
)
IF EXIST "content\models\dpt_large-midas-2f21e586.pt" (
  :: python -m streamlit run scripts\webui_streamlit.py --theme.base dark
  echo all ready to fly

) ELSE (
  echo Your model file does not exist! Place it in 'content\models\dpt_large-midas-2f21e586.pt' with the name 'dpt_large-midas-2f21e586.pt'.
  pause
)