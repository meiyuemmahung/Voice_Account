from flask import Flask, request, jsonify, session, redirect, url_for ,render_template
from flask_cors import CORS
import pyodbc
import secrets 
from utils import *


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 16字節的隨機十六進制字符串
CORS(app)

# 連線資訊
conn_str_master = 'DRIVER={SQL Server};SERVER=MSI;Trusted_Connection=yes;'
conn_str_charge = 'DRIVER={SQL Server};SERVER=MSI;DATABASE=charge;Trusted_Connection=yes;'

# 連接到 master 資料庫
conn_master = pyodbc.connect(conn_str_master)
cursor_master = conn_master.cursor()

# 檢查資料庫是否存在，如果不存在則建立
databases = [db.name for db in cursor_master.execute("SELECT name FROM master.sys.databases").fetchall()]
if 'charge' not in databases:
    # 關閉 master 連線
    conn_master.close()

    # 重新連線到 charge 資料庫
    conn_master = pyodbc.connect(conn_str_master)
    cursor_master = conn_master.cursor()

    # 建立資料庫
    cursor_master.execute("CREATE DATABASE charge")

    # 提交變更
    conn_master.commit()

# 關閉 master 連線
conn_master.close()

# 連接到 charge 資料庫
conn = pyodbc.connect(conn_str_charge)
cursor = conn.cursor()


cursor.execute('''
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'users')
    BEGIN
        CREATE TABLE users (
            uid INT IDENTITY(1,1) PRIMARY KEY,
            name NVARCHAR(255),
            account NVARCHAR(255),
            password NVARCHAR(255)
        )
    END
''')

cursor.execute('''
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'item')
    BEGIN
        CREATE TABLE item (
            id INT IDENTITY(1,1) PRIMARY KEY,
            uid INT FOREIGN KEY REFERENCES users(uid),
            type INT,
            item_name NVARCHAR(255),
            money FLOAT,
            year INT,
            month INT,
            day INT
        )
    END
''')

conn.commit()
cursor, db = db_conn()


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    print(f'Received username: {username}')
    print(f'Received password: {password}')

    cursor.execute('SELECT * FROM users WHERE account = ? AND password = ?', (username, password))
    user = cursor.fetchone()

    if user:
        
        session['username'] = user[1]
        session.permanent = True
        print(f"Login successful for user: {username}")
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'})
    


# 在 /bootstrap_index 路由中檢查當前使用者，如果已登入則返回使用者信息
@app.route('/bootstrap_index')
def bootstrap_index():
    
    username = session.get('username')  # 如果不存在，則返回 None
    return render_template('bootstrap_index.html', username=username)


# 在 /pie_result 路由中檢查當前使用者，如果已登入則返回使用者信息
@app.route('/pie_result')
def pie_result():
    
    username = session.get('username')  # 如果不存在，則返回 None
    return render_template('pie_result.html', username=username)    

# 在 /line_result 路由中檢查當前使用者，如果已登入則返回使用者信息
@app.route('/line_result')
def line_result():
    
    username = session.get('username')  # 如果不存在，則返回 None
    return render_template('line_result.html', username=username)  

@app.route('/overview')
def overview():
    
    username = session.get('username')  # 如果不存在，則返回 None
    return render_template('overview.html', username=username) 

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')

    # 檢查使用者是否已經存在
    cursor.execute('SELECT * FROM users WHERE account = ?', (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({'success': False, 'message': 'Username already exists'})

    # 在users表中新增新使用者
    cursor.execute('INSERT INTO users (name, account, password) VALUES (?, ?, ?)', (name, username, password))
    conn.commit()

    return jsonify({'success': True, 'message': 'Registration successful'})


#登出並清除session
@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logout successful'})



@app.route('/api/get_data_for_month', methods=['POST'])
def get_data_for_month():
    data = request.get_json()
    selected_year = data.get('selectedYear')
    selected_month = data.get('selectedMonth')

    # 使用 session 中的使用者名稱或其他識別符號，這取決於你的資料庫結構
    username = session.get('username')
    
    # 檢索對應的數據
    cursor.execute('''
        SELECT i.type, i.money
        FROM item i
        JOIN users u ON i.uid = u.uid
        WHERE u.name = ? AND i.year = ? AND i.month = ?
    ''', (username, selected_year, selected_month))

    result = cursor.fetchall()
    if not result:
        # 如果結果為空，表示當月無資料
        return jsonify({'message': '當月無資料'})
    # 將數據轉換為字典列表
    data_for_frontend = [{'type': row.type, 'money': row.money} for row in result]

    return jsonify(data_for_frontend)


@app.route('/your_backend_endpoint', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()  # 從請求中獲取 JSON 資料
        received_text = data.get('text', '')

        # 檢查 received_text 是否為空字串
        if not received_text:
            raise ValueError('Received text is empty.')

        # 調用預測函數
        predicted_category = predict_result(received_text)

        item_money  = result(received_text)
        # 分離品項和金額
        item, money = separate_item_and_money(item_money)

        cursor, db = db_conn()
        username = session.get('username')
        cursor.execute('select uid from users where name= convert(nvarchar(max), ?)', username)
        uid = cursor.fetchone()[0]

 
        # 在這裡可以根據你的應用邏輯處理接收到的資料和預測結果
        # 例如，你可能會將 received_text 和 predicted_category 存入資料庫或進行其他操作
        print('接收到的資料:', received_text)
        print('UID:', uid)
        print('預測結果:', predicted_category)
        print('品項:', item)
        print('金額:', money)

        # 加入資料庫
        insert_item(UID=uid, type=predicted_category, item=item, money=money)
        # 如果需要，可以向客戶端發送回應
        return jsonify({'status': 'success', 'received_text': received_text, 'UID':uid, 'predicted_category': predicted_category,'item': item, 'money': money},)

    except Exception as e:
        # 如果有錯誤發生，這裡的代碼就會執行
        error_message = str(e)  # 將錯誤信息轉換為字串
        print('錯誤信息:', error_message)

        # 返回一個包含錯誤信息的 JSON 回應
        return jsonify({'status': 'error', 'error_message': error_message})

@app.route('/overview_item', methods=['POST'])
def overview_item():
    data = request.get_json()
    year = data.get('year')
    month = data.get('month')
    day = data.get('day')

    # 使用 session 中的使用者名稱或其他識別符號，這取決於你的資料庫結構
    username = session.get('username')

    # 檢索對應的數據，包括細項信息
    cursor.execute('''
        SELECT type, SUM(money) as money, item_name
        FROM item
        WHERE uid IN (SELECT uid FROM users WHERE name = ?) AND year = ? AND month = ? AND day = ?
        GROUP BY type, item_name
    ''', (username, year, month, day))

    result = cursor.fetchall()
    if not result:
        # 如果結果為空，表示當天無資料
        return jsonify([])

    # 將數據轉換為字典列表
    data_for_frontend = [{'type': row.type, 'money': row.money, 'item_name': row.item_name} for row in result]

    return jsonify(data_for_frontend)

@app.route('/api/get_six_month_expenses', methods=['GET'])
def get_six_month_expenses():
    try:
        # 使用 session 中的使用者名稱或其他識別符號，這取決於你的資料庫結構
        username = session.get('username')

        # 檢索近六個月的消費金額
        cursor.execute('''
            SELECT year, month, SUM(money) as total_expense
            FROM item
            WHERE uid IN (SELECT uid FROM users WHERE name = ?)
                  AND (YEAR(GETDATE()) - year) * 12 + MONTH(GETDATE()) - month < 6
            GROUP BY year, month
            ORDER BY year DESC, month DESC
        ''', (username,))

        result = cursor.fetchall()

        # 將數據轉換為字典列表
        data_for_frontend = [{'year': row.year, 'month': row.month, 'total_expense': row.total_expense} for row in result]

        return jsonify(data_for_frontend)

    except Exception as e:
        print(f"Error retrieving six-month expenses: {str(e)}")
        return jsonify({'error': 'Unable to retrieve six-month expenses'}), 500




if __name__ == '__main__':
    app.run(debug=True)
