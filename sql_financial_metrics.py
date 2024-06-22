import csv
import os

import mysql.connector

with open("stock_id.txt", "r") as f:
    stock_ids = [line.strip() for line in f]

# Database connection parameters
db_host = "192.168.32.81"
db_user = "root"
db_password = "P@ssw0rd"
db_name = "finindex"

# Establish the connection
connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

# Create a cursor object
cursor = connection.cursor()

# Loop through each stock ID
for stock_id in stock_ids:
    # Define table and CSV file names with current stock ID
    table_name = f"{stock_id}_finindex"
    csv_file = f'C:/Share/期末專題/stock_index/{stock_id}_financial_metrics.csv'

    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' not found for stock {stock_id}.")

        # Drop existing table (if exists)
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create the table with appropriate data types
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                Date DATE PRIMARY KEY NOT NULL,
                Gross_Profit_Margin DOUBLE,
                Operating_Profit_Margin DOUBLE,
                Net_Profit_Margin DOUBLE,
                ROE DOUBLE,
                ROA DOUBLE,
                EPS DOUBLE
            )
        """)

        # Open the CSV file
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')  # Assuming space as delimiter

            # Skip the header row (assuming the first row contains headers)
            next(reader)

            # Prepare the INSERT statement
            insert_query = """
                INSERT INTO {table_name} (DATE, Gross_Profit_Margin, Operating_Profit_Margin, Net_Profit_Margin, ROE, ROA, EPS)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """.format(table_name=table_name)

            # Iterate through the CSV data rows
            for row in reader:
                try:
                    # Convert data types, handle potential conversion errors
                    data_tuple = (row[0], float(row[1] or 0.0), float(row[2] or 0.0), float(row[3] or 0.0),
                                 float(row[4] or 0.0), float(row[5] or 0.0), float(row[6] or 0.0))
                    cursor.execute(insert_query, data_tuple)
                except ValueError:
                    print(f"Error converting data for stock {stock_id}, row {reader.line_num}")
                    continue  # Skip the row with conversion error

        # Commit the changes for the current stock
        connection.commit()
        print(f"Data for stock {stock_id} imported successfully!")

    except FileNotFoundError as e:
        print(e)

# Close the cursor and connection
cursor.close()
connection.close()

print("All stock data import completed!")
