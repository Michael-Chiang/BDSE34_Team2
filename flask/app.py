from flask import Flask, render_template, request, jsonify
import mysql.connector
import time
import datetime

app = Flask(__name__)

# 配置MySQL資料庫連接
db_config = {
    'user': 'root',
    'password': 'P@ssw0rd',
    #'host': '192.168.32.176',
    'host': 'Localhost',
    'database': 'stock',
}

# Grafana 配置
grafana_url = "http://192.168.32.176:3000"
grafana_api_key = "glsa_cNzq1IPXgKS1RqTl1vNPrpcvPAnoOXlt_e278fd0c"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generic')
def generic():
    return render_template('generic.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/filter', methods=['GET'])
def filter():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    
    # 獲取所有的Sector項目
    cursor.execute("SELECT DISTINCT Sector FROM Latest_info")
    sectors = [row['Sector'] for row in cursor.fetchall()]
    
    cursor.close()
    connection.close()
    
    return render_template('filter.html', sectors=sectors)

@app.route('/filter_sectors', methods=['POST'])
def filter_sectors():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    
    sectors = request.form.get('sectors')
    sectors_list = sectors.split(',')

    if sectors_list:
        query = "SELECT DISTINCT Industry FROM Latest_info WHERE Sector IN (%s)" % ','.join(['%s'] * len(sectors_list))
        cursor.execute(query, sectors_list)
    else:
        query = "SELECT DISTINCT Industry FROM Latest_info"
        cursor.execute(query)
    
    industries = [row['Industry'] for row in cursor.fetchall()]
    
    cursor.close()
    connection.close()
    
    return jsonify({'industries': industries})

@app.route('/filter_results', methods=['POST'])
def filter_results():

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    
    sectors = request.form.get('sectors')
    industries = request.form.get('industries')
    page = request.form.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    
    query = "SELECT * FROM Latest_info WHERE 1=1"
    params = []
    
    if sectors:
        sectors_list = sectors.split(',')
        query += " AND Sector IN (%s)" % ','.join(['%s'] * len(sectors_list))
        params.extend(sectors_list)
    
    if industries:
        industries_list = industries.split(',')
        query += " AND Industry IN (%s)" % ','.join(['%s'] * len(industries_list))
        params.extend(industries_list)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    cursor.execute("SHOW COLUMNS FROM Latest_info")
    columns = [column['Field'] for column in cursor.fetchall()]

    cursor.close()
    connection.close()


    return jsonify({'results': results, 'columns': columns})

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    
    query = f"SELECT * FROM `{symbol}`"
    cursor.execute(query)
    results = cursor.fetchall()
    
    cursor.execute(f"SHOW COLUMNS FROM `{symbol}`")
    columns = [column['Field'] for column in cursor.fetchall()]
    
    cursor.close()
    connection.close()

    # 生成 Grafana 仪表板的嵌入 URL
    grafana_dashboard_url = generate_grafana_dashboard_url(symbol)
    
    return render_template('stock_detail.html', results=results, columns=columns, symbol=symbol, grafana_dashboard_url=grafana_dashboard_url)

def generate_grafana_dashboard_url(symbol):
    org_id = 1  
    
    # 獲取當前時間和一個月前的時間
    today = datetime.datetime.now()
    one_month_ago = today - datetime.timedelta(days=90)
    
    from_timestamp = int(time.mktime(one_month_ago.timetuple()) * 1000)
    to_timestamp = int(time.mktime(today.timetuple()) * 1000)
    
    return f"{grafana_url}/d/adompizn6fm68d/a-k-chart?orgId={org_id}&var-StockID={symbol}&from={from_timestamp}&to={to_timestamp}&kiosk"


    

if __name__ == '__main__':
    app.run(debug=True)