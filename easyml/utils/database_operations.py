import sqlite3
from src.utils.logger import write_log
import os
from easyml.config.GLOBAL_VARS import *
import json

def create_database_table(table_name):
    write_log("Entered function create_Database_Table.", "info", os.path.basename(__file__))

    # Connect to the SQLite database with proper string formatting
    conn = sqlite3.connect(f'database//{table_name}.db')
    write_log(f"Created / Connected TrainingData Database - {table_name} table successfully!", "info",
              os.path.basename(__file__))

    cursor_obj = conn.cursor()

    # Dynamically fetch column definitions from global config
    columns_definition = ', '.join(globals()[f"db_{table_name}_columns_config"])

    # Create table
    cursor_obj.execute(f'''CREATE TABLE if not exists {table_name}({columns_definition})''')
    write_log("Train Table created successfully!", "info", os.path.basename(__file__))

    conn.close()

def enter_record_to_table(table_name, **values):
    write_log("Entered function enter_record_to_table.", "info", os.path.basename(__file__))

    # Connect to the appropriate database
    conn = sqlite3.connect(f'database/{table_name}.db')
    cursor_obj = conn.cursor()

    # Prepare columns and values for insertion (excluding 'id')
    columns = ', '.join(values.keys())
    placeholders = ', '.join(['?'] * len(values))
    values = tuple(values.values())

    # Execute the INSERT statement
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    cursor_obj.execute(query, values)

    write_log(f"Inserted record into {table_name}: {values}", "info", os.path.basename(__file__))

    # Commit and close
    conn.commit()
    conn.close()

def show_table():
    write_log("Entered function showTable.", "info", os.path.basename(__file__))
    conn = sqlite3.connect('database/TrainingData.db')
    cursor_obj = conn.cursor()
    output = cursor_obj.execute("select * from Train")

    data = cursor_obj.execute('''SELECT * FROM Train''')
    for column in data.description:
        print(column[0] , end="\t")
    print("")
    for row in output:
        print(row)
    conn.close()

def drop_table(table_name):
    write_log("Entered function drop_table.", "info", os.path.basename(__file__))

    # Connect to the database file dynamically based on table name
    conn = sqlite3.connect(f'database/{table_name}.db')
    cursor_obj = conn.cursor()

    # Drop the table if it exists
    cursor_obj.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()

    write_log(f"Table {table_name} dropped successfully.", "info", os.path.basename(__file__))

def show_all_tables():
    conn = sqlite3.connect('database/TrainingData.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor_obj.fetchall())

def dataframe_to_database(data, table_name="Train"):
    write_log("Entered function dataframe_to_database.", "info", os.path.basename(__file__))

    db_path = f'database/{table_name}.db'
    conn = sqlite3.connect(db_path)

    try:
        data.to_sql(name=table_name, con=conn, index=False, if_exists='replace')
        write_log(f"Data inserted into table '{table_name}' successfully.", "info", os.path.basename(__file__))
    except Exception as e:
        write_log(f"Error writing DataFrame to table '{table_name}': {str(e)}", "error", os.path.basename(__file__))
    finally:
        conn.close()

def get_data_from_database(table_name="runs"):
    write_log("Entered function get_data_from_database.", "info", os.path.basename(__file__))

    db_path = f'database/{table_name}.db'
    conn = sqlite3.connect(db_path)

    try:
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        json_data = {
            "status_code": 200,
            "data": json.loads(data.to_json(orient="records"))
        }
        write_log(f"Data fetched successfully from table '{table_name}'.", "info", os.path.basename(__file__))
        return json_data
    except Exception as e:

        write_log(f"Error fetching data from table '{table_name}': {str(e)}", "error", os.path.basename(__file__))

        return {
            "status_code": 500,
            "error_message": "DatabaseError",
        }

    finally:
        conn.close()

def get_record_by_id(record_id: int, table_name: str = "runs"):
    write_log("Entered function get_record_by_id.", "info", os.path.basename(__file__))

    db_path = f'database/{table_name}.db'
    conn = sqlite3.connect(db_path)
    cursor_obj = conn.cursor()

    try:
        # Query to select record where 'id' is the primary key
        query = f"SELECT * FROM {table_name} WHERE id = ?"
        cursor_obj.execute(query, (record_id,))
        row = cursor_obj.fetchone()

        # Get column names from cursor description
        column_names = [desc[0] for desc in cursor_obj.description]
        result = dict(zip(column_names, row)) if row else None

        if result:
            write_log(f"Record with ID {record_id} fetched successfully from '{table_name}'.", "info", os.path.basename(__file__))
            return result
        else:
            write_log(f"No record found with ID {record_id} in '{table_name}'.", "warning", os.path.basename(__file__))
            raise

    except Exception as e:
        write_log(f"Error retrieving ID {record_id} from table '{table_name}': {str(e)}", "error", os.path.basename(__file__))
        return {
            "status_code": 500,
            "error_message": "DatabaseError"
        }
    finally:
        conn.close()