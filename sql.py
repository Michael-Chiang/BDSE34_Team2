import csv

import mysql.connector

# Database connection parameters
db_host = "192.168.32.81"
db_user = "root"
db_password = "P@ssw0rd"
db_name = "stock"

# Establish the connection
connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

# Create a cursor object
cursor = connection.cursor()

cursor.execute("""drop table if exists Inflation_CP
""")
# Create the table with appropriate data types
cursor.execute("""
CREATE TABLE Inflation_CP (
  Date DATE PRIMARY KEY NOT NULL,
  Inflation_CP DOUBLE)
""")


# Open the CSV file
with open('C:/Share/期末專題/FRED/Inflation, consumer_prices_for_us.csv', 'r') as csvfile:
    # Read the CSV file using a delimiter
    reader = csv.reader(csvfile, delimiter=',')  # Assuming space as delimiter

    # Skip the header row (assuming the first row contains headers)
    next(reader)

    # Prepare the INSERT statement
    insert_query = """
        INSERT INTO Inflation_CP (DATE, Inflation_CP)
        VALUES (%s, %s)
    """

    # Iterate through the CSV data rows
    for row in reader:
        # Convert data types as needed (e.g., convert strings to numbers)
        data_tuple = (row[0], float(row[1] or 0.0))

        # Execute the INSERT statement with the current row data
        cursor.execute(insert_query, data_tuple)

# Commit the changes to the database
connection.commit()  # Ensure to commit changes

# Close the cursor and connection
cursor.close()
connection.close()

print("Inflation_CP data imported successfully!")
