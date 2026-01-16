import psycopg2 # PostgreSQL database adapter for Python

# Establishes a connection to PostgreSQL
# Make sure the database already exists (e.g., signatures_db)
connection = psycopg2.connect(
    host="localhost",
    database="signatures_db",
    user="postgres",
    password="your_password"
)

# Creates a cursor object to interact with the database
cursor = connection.cursor()

# SQL command to create the signatures table
create_table_query = """
CREATE TABLE IF NOT EXISTS signatures (
    id SERIAL PRIMARY KEY,
    person_name VARCHAR(255) NOT NULL,
    image_path TEXT NOT NULL,
    descriptors BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""
# BYTEA is used to store binary data such as image descriptors.

# Executes the SQL command
cursor.execute(create_table_query)

# Saves the changes to the database
connection.commit()
# Closes the cursor and connection
cursor.close()
connection.close()
# Prints a success message
print("Table created successfully.")
