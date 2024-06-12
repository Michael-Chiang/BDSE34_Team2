import csv
import os
import pymysql

# 连接到数据库
conn = pymysql.connect(
    host="192.168.32.81",
    user="root",
    password="P@ssw0rd",
    database="stock",
    autocommit=True,
)
cursor = conn.cursor()

folder_path = "./stock_data"

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        table_name = os.path.splitext(filename)[0]
        csv_file_path = os.path.join(folder_path, filename)

        with open(csv_file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # 读取标题行

            # 打印列名，检查是否有空格或特殊字符
            print("Columns in CSV:", header)

            # 清理列名
            sanitized_columns = [f"`{col.strip()}`" for col in header]
            columns = ", ".join(sanitized_columns)
            values_placeholders = ", ".join(["%s"] * len(header))

            # 删除现有表（如果存在）
            drop_table_statement = f"DROP TABLE IF EXISTS `{table_name}`;"
            cursor.execute(drop_table_statement)

            # 创建表语句，根据CSV文件中的列名创建表
            create_table_statement = f"""
            CREATE TABLE `{table_name}` (
                {', '.join([f'`{col.strip()}` DOUBLE' if col.strip() != 'Date' and col.strip() != 'Volume' else f'`{col.strip()}` INT' if col.strip() == 'Volume' else f'`{col.strip()}` DATE' for col in header])}
            );
            """
            cursor.execute(create_table_statement)

            insert_statement = (
                f"INSERT INTO `{table_name}` ({columns}) VALUES ({values_placeholders})"
            )

            for row in reader:
                cursor.execute(insert_statement, row)

cursor.close()
conn.close()
