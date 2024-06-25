@ECHO OFF

:: 修改console默認encoding为utf8，避免中文亂碼
CHCP 65001

:: 設置環境變數
SET RE=0

SET PYTHON_V=3.11
SET PYTHON_VERSION=3.11.3
SET PYTHON_FILE=python-%PYTHON_VERSION%-amd64.exe
SET PYTHON_DOWNLOAD_URL=https://registry.npmmirror.com/-/binary/python/%PYTHON_VERSION%/%PYTHON_FILE%
SET PYTHON_FULL_VERSION=python-%PYTHON_VERSION%

SET GIT_VERSION=2.40.1
SET GIT_FILE=Git-%GIT_VERSION%-64-bit.exe
SET GIT_DOWNLOAD_URL=https://registry.npmmirror.com/-/binary/git-for-windows/v%GIT_VERSION%.windows.1/%GIT_FILE%
SET GIT_FULL_VERSION=git-%GIT_VERSION%

%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
cd /d "%~dp0"

:: 檢查是否安裝了python3.11
py -%PYTHON_V% --version  >nul 2>nul
if %errorlevel%==0 (echo Python %PYTHON_V% 運行正常！) else (
  echo Python %PYTHON_V% 運行異常！開始下載 %PYTHON_FULL_VERSION%......請耐心等待 
  ::-----安装python
  :: 如果已存在則删除
  if exist %PYTHON_FILE% del %PYTHON_FILE%
  :: 下載python
  powershell -c "invoke-webrequest -uri %PYTHON_DOWNLOAD_URL% -outfile %PYTHON_FILE%"
  :: 判斷是否下載完成
  if not exist %PYTHON_FILE% cls&color 0c&echo [Error]:下載python失敗!...&pause>nul&exit
  echo.
  echo 下載完成，正在後台安裝 %PYTHON_FULL_VERSION%

  %PYTHON_FILE%  /quiet PrependPath=1
  del %PYTHON_FILE%

  SET RE=1
)

::檢查是否安裝了git
git --version >nul 2>nul
if %errorlevel%==0 (echo %GIT_FULL_VERSION% 運行正常！) else (
  echo %GIT_FULL_VERSION% 運行異常！開始下載 %GIT_FULL_VERSION%......請耐心等待 
  ::-----安装git
  :: 如果已存在則删除
  if exist %GIT_FILE% del %GIT_FILE%
  :: 下載 git
  powershell -c "invoke-webrequest -uri %GIT_DOWNLOAD_URL% -outfile %GIT_FILE%"
  :: 判斷是否下載完成
  if not exist %GIT_FILE% cls&color 0c&echo [Error]:下載git失敗!...&pause>nul&exit
  echo.
  echo 下載完成，正在後台安裝 %GIT_FULL_VERSION%

  %GIT_FILE%  /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"
  del %GIT_FILE%

  SET RE=1
)

if %RE%==1 (
  echo.
  echo python 和 git 安装完成，請手動重啟本 bat 文件！
  pause
  exit
)

::更新github pull
::檢查是否含有Django項目文件夾
if exist django (
  cd /d "%~dp0django"
) else (
  git clone -b bdse34/team2/joanna8799/django --single-branch https://github.com/dennisNism/BDSE34_Team2_20.git django --recursive
  if exist django (
    cd /d "%~dp0django"
  ) else (
    echo github 连接失败，请再次尝试
    pause
    exit
  )
)
echo.
echo 正在從remote端更新
echo %~dp0django
git fetch origin
git checkout bdse34/team2/joanna8799/django
git pull origin bdse34/team2/joanna8799/django --recurse-submodules
echo.
echo 更新完成！


::檢查虛擬環境
if exist venv (
  call venv\Scripts\activate.bat
  if %errorlevel%==0 (
    echo.
    echo 已進入 venv 虛擬環境
  )
) else (
  echo 未創建 venv 虛擬環境，正在創建！
  python -m venv venv 
  call venv\Scripts\activate.bat
  echo.
  echo 已進入 venv 虛擬環境
)

::安裝pip
echo.
echo 正在檢查並安裝 pip 環境包

python.exe -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo pip 環境包更新完成

::遷移資料庫
echo.
echo 正在遷移資料庫...

python manage.py migrate

::啟動
echo.
echo 正在啟動 Django ...

python manage.py runserver

pause

