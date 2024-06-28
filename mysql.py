import csv
import os

import pymysql
# from dotenv import get_key


def import_csv_to_mysql(host, user, password, database, folder_path):
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            autocommit=True,
        )
        cursor = conn.cursor()

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                table_name = os.path.splitext(filename)[0]
                csv_file_path = os.path.join(folder_path, filename)
                with open(csv_file_path, "r") as f:
                    reader = csv.reader(f)
                    print(f"Processing file: {filename}")

                    header = next(reader)
                    print("Columns in CSV:", header)

                    sanitized_columns = [f"`{col.strip()}`" for col in header]
                    columns = ", ".join(sanitized_columns)
                    values_placeholders = ", ".join(["%s"] * len(header))

                    drop_table_statement = f"DROP TABLE IF EXISTS `{table_name}`;"
                    cursor.execute(drop_table_statement)

                    create_table_statement = f"""
                    CREATE TABLE `{table_name}` (
                        {', '.join([f'`{col.strip()}` DOUBLE' if col.strip() != 'Date' and col.strip() != 'Volume' else f'`{col.strip()}` BIGINT' if col.strip() == 'Volume' else f'`{col.strip()}` DATE' for col in header])}
                    );
                    """
                    cursor.execute(create_table_statement)

                    insert_statement = f"INSERT INTO `{table_name}` ({columns}) VALUES ({values_placeholders})"

                    for row in reader:
                        cursor.execute(insert_statement, row)

        cursor.close()
        conn.close()
        print("Data import completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# password = get_key(".env", "password")


if __name__ == "__main__":
    import_csv_to_mysql(
        host="192.168.32.176",
        user="root",
        password='P@ssw0rd',
        database="stock",
        folder_path="stock_data_model",
    )
