### 基於django框架製作的股票篩選網頁
+ 串接 MySQL 資料庫與 grafana server，請確保他們開啟服務
+ releases 有 django_run.bat 一鍵運行包，建立虛擬環境並安裝套件 (如果你使用源碼執行，請忽略他)
+ 環境/套件：請依需求自行創建虛擬環境，並安裝 requirements.txt 文件內的套件，或手動執行：
`pip install django mysqlclient`
+ 開啟測試運行 server 方式：
1. 切換到src目錄
`cd django\src`
2. 執行python manage.py runserver 指令
`python manage.py runserver`
