import os
import shutil
import sqlite3
from contextlib import closing

from easyml.config.constants import (
    db_recipes_columns_config,
    db_runs_columns_config,
    db_schedule_events_columns_config,
    db_schedules_columns_config,
)
from easyml.config.paths import DB_DIR
from easyml.utils.logger import write_log

# Use centralized database directory
db_dir = DB_DIR

_TABLE_COLUMNS = {
    "runs": db_runs_columns_config,
    "recipes": db_recipes_columns_config,
    "schedules": db_schedules_columns_config,
    "schedule_events": db_schedule_events_columns_config,
}


def _column_names(table_name):
    return {definition.split()[0] for definition in _TABLE_COLUMNS[table_name]}


def update_record(table_name, record_id, **values):
    """Set ``values`` on one row. Only the table's declared columns can be written."""
    unknown = set(values) - _column_names(table_name) - {"id"}
    if unknown or "id" in values:
        raise ValueError(f"Cannot update column(s) {', '.join(sorted(unknown | ({'id'} & set(values))))}")
    if not values:
        return
    assignments = ", ".join(f"{name} = ?" for name in values)
    with closing(_connect(table_name)) as conn:
        conn.execute(f"UPDATE {table_name} SET {assignments} WHERE id = ?", (*values.values(), record_id))
        conn.commit()


def claim_record(table_name, record_id, condition, params=(), **values):
    """
    Set ``values`` on one row only if ``condition`` (internal SQL, never user input) holds, atomically.
    Returns True when this caller changed the row, so two processes can never both claim it.
    """
    unknown = set(values) - _column_names(table_name)
    if unknown or not values:
        raise ValueError(f"Cannot update column(s) {', '.join(sorted(unknown))}")
    assignments = ", ".join(f"{name} = ?" for name in values)
    with closing(_connect(table_name)) as conn:
        cursor = conn.execute(f"UPDATE {table_name} SET {assignments} WHERE id = ? AND ({condition})",
                              (*values.values(), record_id, *params))
        conn.commit()
        return cursor.rowcount == 1


def delete_record(table_name, record_id):
    """Delete one row (for tables without files on disk; runs use delete_run_by_id)."""
    with closing(_connect(table_name)) as conn:
        conn.execute(f"DELETE FROM {table_name} WHERE id = ?", (record_id,))
        conn.commit()


class RunNotFoundError(LookupError):
    """Raised when a run id does not exist in the runs table."""


def _connect(table_name):
    if table_name not in _TABLE_COLUMNS:
        raise ValueError(f"Unknown table '{table_name}'")
    return sqlite3.connect(os.path.join(db_dir, f"{table_name}.db"))


def create_database_table(table_name):
    """
    Creates a SQLite table if it doesn't exist using predefined column configs.

    Args:
        table_name (str): Name of the table to create.
    """

    write_log("Entered function create_Database_Table.", "debug", os.path.basename(__file__))

    columns_definition = ', '.join(_TABLE_COLUMNS[table_name])
    with closing(_connect(table_name)) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name}({columns_definition})")
        _add_missing_columns(conn, table_name)
        conn.commit()

    write_log(f"Created / Connected {table_name} table successfully!", "debug", os.path.basename(__file__))


_ENSURED = set()


def ensure_table(table_name):
    """Create the table if needed, without logging (for code that runs every few seconds, like the scheduler)."""
    columns_definition = ', '.join(_TABLE_COLUMNS[table_name])
    with closing(_connect(table_name)) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name}({columns_definition})")
        if table_name not in _ENSURED:  # migrate older databases once per process
            _add_missing_columns(conn, table_name)
            _ENSURED.add(table_name)
        conn.commit()


def fetch_rows(table_name, where="", params=(), order="id", limit=None):
    """Rows as dicts. ``where`` and ``order`` are internal SQL fragments, never user input; values go in ``params``."""
    ensure_table(table_name)
    query = f"SELECT * FROM {table_name}" + (f" WHERE {where}" if where else "") + f" ORDER BY {order}"
    if limit:
        query += f" LIMIT {int(limit)}"
    with closing(_connect(table_name)) as conn:
        cursor = conn.execute(query, tuple(params))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _add_missing_columns(conn, table_name):
    """Bring a database created by an older version up to the current column list."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    for definition in _TABLE_COLUMNS[table_name]:
        name = definition.split()[0]
        if name in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {definition}")
            write_log(f"Added column {name} to {table_name}", "info", os.path.basename(__file__))
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():  # another process migrated first
                raise


def enter_record_to_table(table_name, **values):
    """
    Inserts a record into the specified table.

    Args:
        table_name (str): Target table name.
        **values: Column-value pairs to insert.

    Returns:
        int: The id of the inserted row.
    """

    write_log("Entered function enter_record_to_table.", "debug", os.path.basename(__file__))

    columns = ', '.join(values.keys())
    placeholders = ', '.join(['?'] * len(values))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    with closing(_connect(table_name)) as conn:
        cursor_obj = conn.execute(query, tuple(values.values()))
        conn.commit()
        row_id = cursor_obj.lastrowid

    write_log(f"Inserted record {row_id} into {table_name}", "info", os.path.basename(__file__))
    return row_id


def get_data_from_database(table_name="runs"):
    """
    Fetches all rows from the specified table.

    Args:
        table_name (str): The table to query.

    Returns:
        dict: Contains status code and data or error message.
    """

    write_log("Entered function get_data_from_database.", "debug", os.path.basename(__file__))

    try:
        with closing(_connect(table_name)) as conn:
            cursor = conn.execute(f"SELECT * FROM {table_name} ORDER BY id")
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        write_log(f"Data fetched successfully from table '{table_name}'.", "debug", os.path.basename(__file__))
        return {
            "status_code": 200,
            "data": data,
        }
    except Exception as e:
        write_log(f"Error fetching data from table '{table_name}': {str(e)}", "error", os.path.basename(__file__))
        return {
            "status_code": 500,
            "error_message": "DatabaseError",
        }


def folder_has_run(folder_path, table_name="runs"):
    """Return True if a run record points at ``folder_path``."""
    target = os.path.normcase(os.path.realpath(folder_path))
    with closing(_connect(table_name)) as conn:
        rows = conn.execute(f"SELECT folder_path FROM {table_name}").fetchall()
    return any(path and os.path.normcase(os.path.realpath(path)) == target for (path,) in rows)


def get_record_by_id(record_id: int, table_name: str = "runs"):
    """
    Fetches a record from the specified table using its ID.

    Args:
        record_id (int): ID of the row.
        table_name (str): Table name to query.

    Returns:
        dict: The record as a dictionary.

    Raises:
        RunNotFoundError: If no row has this id.
    """

    write_log("Entered function get_record_by_id.", "debug", os.path.basename(__file__))

    with closing(_connect(table_name)) as conn:
        cursor_obj = conn.execute(f"SELECT * FROM {table_name} WHERE id = ?", (record_id,))
        row = cursor_obj.fetchone()
        column_names = [desc[0] for desc in cursor_obj.description]

    if row is None:
        write_log(f"No record found with ID {record_id} in '{table_name}'.", "warning", os.path.basename(__file__))
        raise RunNotFoundError(f"Run with ID {record_id} was not found.")

    write_log(f"Record with ID {record_id} fetched successfully from '{table_name}'.", "debug", os.path.basename(__file__))
    return dict(zip(column_names, row))


def delete_run_by_id(run_id: int, table_name: str = "runs"):
    """
    Deletes a training run from the database and its associated folder.

    Args:
        run_id (int): Run ID to delete.
        table_name (str): Table to delete from.

    Returns:
        dict: Result with status code and message.
    """

    write_log("Entered function delete_run_by_id.", "debug", os.path.basename(__file__))

    try:
        with closing(_connect(table_name)) as conn:
            result = conn.execute(f"SELECT folder_path FROM {table_name} WHERE id = ?", (run_id,)).fetchone()

            if not result:
                write_log(f"No run found with ID {run_id} in table '{table_name}'.", "warning", os.path.basename(__file__))
                return {
                    "status_code": 404,
                    "message": f"Run with ID {run_id} not found."
                }

            folder_path = result[0]
            if folder_path:
                from easyml.core.model_cache import forget_folder

                forget_folder(folder_path)
            if folder_path and os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                write_log(f"Deleted folder at path: {folder_path}", "info", os.path.basename(__file__))
            else:
                write_log(f"Folder path not found or doesn't exist: {folder_path}", "warning", os.path.basename(__file__))

            conn.execute(f"DELETE FROM {table_name} WHERE id = ?", (run_id,))
            conn.commit()

        write_log(f"Run with ID {run_id} deleted successfully from '{table_name}'.", "info", os.path.basename(__file__))
        return {
            "status_code": 200,
            "message": f"Run with ID {run_id} and associated folder deleted successfully."
        }

    except Exception as e:
        write_log(f"Error deleting run with ID {run_id} from '{table_name}': {str(e)}", "error", os.path.basename(__file__))
        return {
            "status_code": 500,
            "error_message": "DatabaseError"
        }
