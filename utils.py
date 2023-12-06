import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from transformers import (
    BertModel, 
    BertTokenizer, 
    BertForSequenceClassification
)

import numpy as np
import pandas as pd
import math

import pyodbc
import pandas as pandas
pyodbc.drivers()
from datetime import datetime

from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok # colab
from llama_cpp import Llama
import json
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ______________________________________________________________________________
# Functions on Sequences and Iterables

class BertClassifier(nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-chinese')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, len(labels))
    self.relu = nn.ReLU()

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    final_layer = self.relu(linear_output)
    return final_layer
  
def predict_result(new_text, model_path='C:/Users/abcde/Desktop/account/bert_model .pth'):
    # 載入模型
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 確定運行裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用與訓練時相同的tokenizer進行tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    encoded_text = tokenizer.encode_plus(new_text,
                        padding='max_length',
                        max_length=512,
                        truncation=True,
                        return_tensors="pt")

    # 將 encoded_text 移動到相同的裝置上
    encoded_text = {key: value.to(device) for key, value in encoded_text.items()}

    # 使用模型進行預測
    model.eval()
    with torch.no_grad():
      output = model(encoded_text['input_ids'], encoded_text['attention_mask'])

    # 使用 softmax 函數獲得每一類的概率值
    probabilities = torch.softmax(output, dim=1).squeeze().tolist()

    # 定義類別到標籤的映射
    category_mapping = {0: '食', 1: '衣', 2: '住', 3: '行', 4: '育', 5: '樂', 6: '其他'}

    # 找到概率低於50%的類別
    low_prob_categories = [i for i, prob in enumerate(probabilities) if prob < 0.5]

    # 如果所有類別的概率都低於50%，將其歸為 "其他" 類
    if len(low_prob_categories) == len(category_mapping) - 1:  # 減去 "其他" 類
      predicted_category_label = '其他'
    else:
      # 取得最大概率對應的類別
      predicted_category = torch.argmax(output, dim=1).item()
      predicted_category_label = category_mapping[predicted_category]

    # # 打印每一類的預測概率
    # print("預測概率：")
    # for i, prob in enumerate(probabilities):
    #   print(f"{category_mapping[i]}: {prob * 100:.2f}%")

    # # 打印最終的預測結果
    # print(f"預測結果: {predicted_category_label}")

    # 返回最終的預測結果
    return predicted_category


# ______________________________________________________________________________
# SQL


def db_conn(server='MSI', database='charge', username='', password=''):
    try:
      db = pyodbc.connect(f'Driver=ODBC Driver 17 for SQL Server;'
                          f'Server={server};'
                          f'database={database};'
                          f'UID={username};'
                          f'PWD={password};'
                          f'Trusted_Connection=yes;')
      cursor = db.cursor()
      return cursor, db
    except Exception as e:
      print(f"Error: {e}")
      return None, None
cursor, db = db_conn()

def insert_user(cursor, name, account, password): # insert new data to users
    # 建立連接
    cursor, db = db_conn()
    # 插入一條新記錄到 users 表格
    try:
        cursor.execute('INSERT INTO users (Name, account, password) VALUES (?, ?, ?)', name, account, password)
        # 提交更改
        cursor.commit()

        print("已創建成功！")
    except pyodbc.IntegrityError as e:
        print("此帳號已被註冊")
    except Exception as e:
        print(f"發生未知錯誤：{e}")
    finally:
        # 關閉連接
        cursor.close()
        db.close()


def insert_item(UID, type, item, money):
    try:
        # 建立連接
        cursor, db = db_conn()
        # 獲取當前日期和時間
        current_datetime = datetime.now()

        # 提取年份、月份和日
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day

        # 檢查 item 和 money 是否為 None 或 NaN，如果是，則不進行插入
        if item is not None and not math.isnan(money):
            # 插入一條新交易記錄到 item 表格
            cursor.execute('INSERT INTO item (UID, Type, Item_name, Money, Year, Month, Day) VALUES (?, ?, ?, ?, ?, ?, ?)',
                           UID, type, item, money, year, month, day)
            
            # 提交更改
            db.commit()
            print("成功插入資料")
        else:
            print("Item 或 Money 為 None 或 NaN, 未進行插入")
    except Exception as e:
        print("插入資料失敗:", e)
    finally:
        # 關閉連接
        cursor.close()
        db.close()


def delete_users(cursor, uid):
    try:
        # 使用 db_conn 函數建立連接
        cursor, db = db_conn()
        # 使用 cursor 執行 SQL 刪除特定行的語句
        cursor.execute('DELETE FROM users WHERE UID = ?', uid)
        # 提交更改
        cursor.commit()
    except Exception as e:
        # 發生異常時的處理
        print(f"刪除會員時發生錯誤：{e}")
        # 如果發生異常，可能需要回滾事務
        if db:
            db.rollback()
    finally:
        # 無論是否發生異常，都需要關閉資料庫連接
        if db:
            db.close()

def delete_item(cursor, item_id):
    try:
        # 使用 db_conn 函數建立連接
        cursor, db = db_conn()
        # 使用 cursor 執行 SQL 刪除特定行的語句
        cursor.execute('DELETE FROM item WHERE ID = ?', item_id)
        # 提交更改
        cursor.commit()
    except Exception as e:
        # 發生異常時的處理
        print(f"刪除時發生錯誤：{e}")
        # 如果發生異常，可能需要回滾事務
        if db:
            db.rollback()
    finally:
        # 無論是否發生異常，都需要關閉資料庫連接
        if db:
            db.close()


def db_query_to_dataframe(cursor, table_name):
    try:
        # 使用 db_conn 函數建立連接
        cursor, db = db_conn()
        query = f'SELECT * FROM {table_name}'
        # 使用 cursor 執行 SQL 查詢
        cursor.execute(query)
        # 取得查詢結果
        rows = cursor.fetchall()
        # 取得查詢結果的欄位名稱
        columns = [column[0] for column in cursor.description]
        # 將查詢結果轉換為 DataFrame
        df = pd.DataFrame.from_records(rows, columns=columns)
        return df
    except pyodbc.ProgrammingError as e:
        # 表不存在的異常處理
        print(f"查詢資料表時發生錯誤：{e}")
        # 在這裡您可以選擇返回一個空的 DataFrame 或者其他預設值
        return pd.DataFrame()
    except Exception as e:
        # 其他異常的處理
        print(f"發生未知錯誤：{e}")
    finally:
        # 無論是否發生異常，都需要關閉資料庫連接
        if db:
            db.close()


# ______________________________________________________________________________
# llama2

llm = Llama(model_path="C:/Users/abcde/Desktop/account/ckip-llama27bchat-q8.gguf.gguf", n_gpu_layers=12)

def result(received_text):
    QA = f"你現在是提取句子中特徵的達人,我會輸入一句話，請提取句子中的品項及金額。 我給你以下的範例, 請試著學習: \
        HUMAN: 例句: 我今天吃了一碗拉麵花了60元 \
            結果: 拉麵, 60 \
            例句: 我今天晚餐消費700元 \
            結果: 晚餐, 700 \
            例句: 我給自己換了一張電競椅價格是15000塊 \
            結果: 電競椅, 15000 \
            例句: 每年的汽車保險花費9000元 \
            結果: 汽車保險, 9000 \
            問題: {received_text}。請按照範例輸出結果。 \
        ASSISTANT: "

    out = llm(QA, max_tokens=2048, temperature=0.5, echo=True)
    out = out['choices'][0]['text'].split('ASSISTANT:')[1]
    if not out:
        out = "No comment!"
    return str(out)


# 假設您的品項和金額分離的程式碼
def separate_item_and_money(text):
    if '結果:' not in text:
        match = re.match(r'(.+?),\s(\d+)', text)
        item = match.group(1).strip()
        money = int(match.group(2))
        return item, money
    else:
        match = re.search(r'結果:\s(.+)', text)
        if match:
            result = match.group(1).strip()

            parts = result.split(',')
            if len(parts) == 2:
                item = parts[0].strip()
                money = int(parts[1].strip())
                return item, money
            else:
                return None, None
        else:
            return None, None
