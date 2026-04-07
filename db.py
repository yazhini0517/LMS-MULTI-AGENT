import psycopg2

def get_connection():
    conn = psycopg2.connect(
        dbname="lms_db",
        user="postgres",
        password="my_password",
        host="localhost",
        port="55432"
    )
    return conn