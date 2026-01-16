from src.db_utils import get_connection 
# Establishes a connection to PostgreSQL
conn = get_connection() 
print("Connected successfully!") 
# Closes the connection
conn.close()