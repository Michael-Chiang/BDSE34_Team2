from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection
import time
import datetime

# Create your views here.




# Grafana 配置
grafana_url = "http://192.168.32.176:3000"
grafana_api_key = "glsa_cNzq1IPXgKS1RqTl1vNPrpcvPAnoOXlt_e278fd0c"


def index(request):
    # 這個視圖函數將渲染 html 模板
    return render(request, 'index.html')

def filter(request):
    """
    獲取所有的Sector項目
    """
    sectors = []
    with connection.cursor() as cursor:
        
        cursor.execute("SELECT DISTINCT Sector FROM general_info")
        sectors = [row[0] for row in cursor.fetchall()]
        
    return render(request, 'filter.html', {'sectors': sectors})

def filter_sectors(request):
    """
    根據sectors篩選取得所有的Industry項目
    """
    sectors = request.POST.get('sectors', '').split(',')
    industries = []

    if sectors:
        query = "SELECT DISTINCT Industry FROM general_info WHERE Sector IN (%s)" % ','.join(['%s'] * len(sectors))
        with connection.cursor() as cursor:
            cursor.execute(query, sectors)
            industries = [row[0] for row in cursor.fetchall()]
            print(industries)
    else:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Industry FROM general_info")
            industries = [row[0] for row in cursor.fetchall()]
    
    return JsonResponse({'industries': industries})

def filter_results(request):
    """
    獲取篩選後的所有股票信息
    """
    sectors = request.POST.get('sectors')
    industries = request.POST.get('industries')
    
    query = "SELECT * FROM general_info WHERE 1=1"
    params = []
    
    if sectors:
        sectors_list = sectors.split(',')
        query += " AND Sector IN (%s)" % ','.join(['%s'] * len(sectors_list))
        params.extend(sectors_list)
    
    if industries:
        industries_list = industries.split(',')
        query += " AND Industry IN (%s)" % ','.join(['%s'] * len(industries_list))
        params.extend(industries_list)
    
    results = []
    columns = []
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

    results_dict = [dict(zip(columns, row)) for row in results]
    

    # 定義想要保留的欄位名稱
    desired_columns = ['Symbol', 'CorName', 'Sector', 'Industry']  

    # 去除不需要的欄位
    filtered_results = [{k: v for k, v in row.items() if k in desired_columns} for row in results_dict]


    
    return JsonResponse({'results': filtered_results, 'columns': desired_columns})

def stock_detail(request, symbol):
    """
    獲取所有的Sector項目
    """
    query = f"SELECT * FROM `{symbol}`"
    results = []
    columns = []

    with connection.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

    results_dict = [dict(zip(columns, row)) for row in results]
    # 将结果转换为列表格式，便于模板解析
    results_list = [list(row) for row in results]
    
    
    grafana_dashboard_url = generate_grafana_dashboard_url(symbol)
    
    return render(request, 'stock_detail.html', {'results': results_list, 'columns': columns, 'symbol': symbol, 'grafana_dashboard_url': grafana_dashboard_url})

def generate_grafana_dashboard_url(symbol):
    """
    獲取所有的Sector項目
    """
    org_id = 1  
    today = datetime.datetime.now()
    one_month_ago = today - datetime.timedelta(days=90)
    
    from_timestamp = int(one_month_ago.timestamp() * 1000)
    to_timestamp = int(today.timestamp() * 1000)
    
    return f"http://192.168.32.176:3000/d/adompizn6fm68d/a-k-chart?orgId={org_id}&var-StockID={symbol}&from={from_timestamp}&to={to_timestamp}&kiosk"