from flask import  render_template, request, jsonify
import mysql.connector
import time
import datetime


# 配置MySQL資料庫連接
db_config = {
    'user': 'root',
    'password': 'P@ssw0rd',
    #'host': '192.168.32.176', # MySQL
    'host': 'Localhost', # MySQL本地
    #'host': 'StockDB', # Docker MySQL
    'database': 'stock',
}

# Grafana 配置
grafana_url = "http://192.168.32.176:3000"
grafana_api_key = "glsa_cNzq1IPXgKS1RqTl1vNPrpcvPAnoOXlt_e278fd0c"

def init_route(app):
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

        # cursor.execute("SELECT MIN(`Latest Price`) as min_price, MAX(`Latest Price`) as max_price FROM Latest_info")
        # price_range = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return render_template('filter.html', 
                            sectors=sectors, 
                            #price_range=price_range
                            )

    def value_to_text(value):
        text_map = {
            0 : '大跌',
            1 : '小跌',
            2 : '持平',
            3 : '小漲',
            4 : '大漲'
        }
        return text_map.get(value, '')

    @app.route('/prediction', methods=['GET'])
    def prediction():
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        # 只获取 Finance 和 Technology 的 Sector 项目
        cursor.execute("SELECT DISTINCT Sector FROM Latest_info WHERE Sector IN ('Finance', 'Technology')")
        sectors = [row['Sector'] for row in cursor.fetchall()]

        # cursor.execute("SELECT MIN(`Latest Price`) as min_price, MAX(`Latest Price`) as max_price FROM Latest_info")
        # price_range = cursor.fetchone()
        
        cursor.execute("SELECT `DL Prediction` FROM (SELECT DISTINCT `DL Prediction` FROM Latest_info WHERE `Sector` IN ('Finance', 'Technology')) AS subquery ORDER BY `DL Prediction` ASC;")
        dl_rows = cursor.fetchall()
        prediction_dl_values = [int(row['DL Prediction']) for row in dl_rows]

        cursor.execute("SELECT `ML Prediction` FROM (SELECT DISTINCT `ML Prediction` FROM Latest_info WHERE `Sector` IN ('Finance', 'Technology')) AS subquery ORDER BY `ML Prediction` ASC;")
        ml_rows = cursor.fetchall()
        prediction_ml_values = [int(row['ML Prediction']) for row in ml_rows]
        print(prediction_ml_values)
        
        
        cursor.close()
        connection.close()

        prediction_dl_texts = [value_to_text(value) for value in prediction_dl_values]
        prediction_ml_texts = [value_to_text(value) for value in prediction_ml_values]
        
        return render_template('prediction.html', 
                            sectors=sectors, 
                            #price_range=price_range, 
                            prediction_dl_values=prediction_dl_values, 
                            prediction_ml_values=prediction_ml_values,
                            prediction_dl_texts=prediction_dl_texts,
                            prediction_ml_texts=prediction_ml_texts,
                            )




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

    @app.route('/filter_results', methods=['POST', 'GET'])
    def filter_results():

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        sectors = request.form.get('sectors')
        industries = request.form.get('industries')
        stock_id = request.form.get('stock_id')
        min_price = request.form.get('min_price', type=float)
        max_price = request.form.get('max_price', type=float)
        prediction_dl = request.form.get('prediction_dl')
        prediction_ml = request.form.get('prediction_ml')
        page_type = request.form.get('page_type')

        
        # 添加过滤条件
        if page_type == 'prediction':
            query = "SELECT * FROM Latest_info WHERE `Sector` IN ('Finance', 'Technology') AND 1=1"

        else:
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

        if stock_id:  
            query += " AND Symbol LIKE %s"
            params.append('%' + stock_id + '%')

        if min_price is not None:
            query += " AND `Latest Price` >= %s"
            params.append(min_price)

        if max_price is not None:
            query += " AND `Latest Price` <= %s"
            params.append(max_price)

        if prediction_dl:
            prediction_dl_list = prediction_dl.split(',')
            query += " AND `DL Prediction` IN (%s)" % ','.join(['%s'] * len(prediction_dl_list))
            params.extend(prediction_dl_list)

        if prediction_ml:
            prediction_ml_list = prediction_ml.split(',')
            query += " AND `ML Prediction` IN (%s)" % ','.join(['%s'] * len(prediction_ml_list))
            params.extend(prediction_ml_list)
        
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        cursor.execute("SHOW COLUMNS FROM Latest_info")
        columns = [column['Field'] for column in cursor.fetchall()]

        cursor.execute("SELECT MIN(`Latest Price`) as min_price, MAX(`Latest Price`) as max_price FROM Latest_info")
        price_range = cursor.fetchone()

        cursor.close()
        connection.close()


        return jsonify({'results': results, 'columns': columns, 'price_range': price_range})

    @app.route('/get_price_range', methods=['GET'])
    def get_price_range():
        page_type = request.args.get('page_type')
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        

        if page_type == 'prediction':
            cursor.execute("SELECT MIN(`Latest Price`) as min_price, MAX(`Latest Price`) as max_price FROM Latest_info WHERE Sector IN ('Finance', 'Technology')")      
            price_range = cursor.fetchone()
            
        else:
            cursor.execute("SELECT MIN(`Latest Price`) as min_price, MAX(`Latest Price`) as max_price FROM Latest_info")
            price_range = cursor.fetchone()

        cursor.close()
        connection.close()
        
        return jsonify(price_range)

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
        

