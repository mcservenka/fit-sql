import os
import json
import sqlite3
from configs.paths import SCHEMAS_PATH, SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH

class SchemaBuilder:

    """
    Builds schema representation based on db_id and dataset
    Stores representation as json file and loads it from json
    Generates schema string based on json structure
    """


    def __init__(self, dataset:str="spider", db_id:str=None):

        self.dataset = dataset
        self.db_id = db_id
        self.reset()

        if dataset == "spider":
            self.db_path = f"{SPIDER_DATABASE_PATH}{db_id}/{db_id}.sqlite"
        elif dataset == "bird":
            self.db_path = f"{BIRD_DATABASE_PATH}{db_id}/{db_id}.sqlite"

        os.makedirs(SCHEMAS_PATH, exist_ok=True)
        os.makedirs(f"{SCHEMAS_PATH}{dataset}", exist_ok=True)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    
    
    #  establish sqlite connection
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    # close sqlite connection
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cursor = None

    # reset all attributes
    def reset(self):
        self.conn = None
        self.cursor = None
        self.tables = []
        self.primary_keys = {}
        self.foreign_keys = {}
        self.columns = {}
        self.schema_object = None


    
    def _get_tables(self):

        self.tables = [] # reset tables

        sql = """ SELECT name 
                  FROM sqlite_master 
                  WHERE type='table' 
                  AND name NOT LIKE 'sqlite_%';
              """
        self.cursor.execute(sql)
        tables = to_dict(cursor=self.cursor)
        
        if tables:
            self.tables = [table["name"] for table in tables]
        
    def _get_primary_keys(self):

        self.primary_keys = {} # reset primary keys
        if len(self.tables) == 0:
            raise ValueError("Tables must not be empty for primary key extraction.")
        
        pks = {}

        for table in self.tables:
            try:
                sql = f"PRAGMA table_info([{table}])"
                self.cursor.execute(sql)
                info = to_dict(cursor=self.cursor)
                
                for row in info:
                    if row.get("pk") > 0:
                        pks.setdefault(table, []).append(row.get("name"))
            except Exception as e:
                print(f"db_id: {self.db_id}")
                print(f"current table: {table}")
                raise Exception(e)
        
        self.primary_keys = pks

    def _get_foreign_keys(self):
        
        self.foreign_keys = {} # reset foreign keys
        if len(self.tables) == 0:
            raise ValueError("Tables must not be empty for foreign key extraction.")

        fks = {}

        for table in self.tables:

            sql = f"PRAGMA foreign_key_list([{table}]);"                
            self.cursor.execute(sql)
            info = to_dict(cursor=self.cursor)

            if len(info) > 0:
                for row in info:
                    fks.setdefault(table, []).append({"sourceTable": row.get("table"),
                                                        "sourceColumn": row.get("to"),
                                                        "targetColumn": row.get("from")})
        
        self.foreign_keys = fks

    def _get_columns(self):

        self.columns = {} # reset columns
        if len(self.tables) == 0:
            raise ValueError("Tables must not be empty for column extraction.")        

        cols = {}

        for table in self.tables:
            sql = f"PRAGMA table_info([{table}])"
            self.cursor.execute(sql)
            info = to_dict(cursor=self.cursor)
            
            for row in info:
                cols.setdefault(table, []).append({"name": row.get("name"),
                                                    "type": row.get("type"),
                                                    "typegroup": normalize_type(row.get("type")),
                                                    "notnull": bool(row.get("notnull")),
                                                    "pk": row.get("pk")})
        
        self.columns = cols
    
    def build_schema_object(self):
        self._get_tables()
        self._get_primary_keys()
        self._get_foreign_keys()
        self._get_columns()

        obj = {"dataset": self.dataset, "db_id": self.db_id, "schema": {}}
        for table in self.tables:
            obj["schema"][table] = {
                "columns": self.columns.get(table, []),
                "primary_keys": self.primary_keys.get(table, []),
                "foreign_keys": self.foreign_keys.get(table, [])
            }
        self.schema_object = obj
        return obj

    # save schema object as json
    def save_schema_json(self):
        if not self.schema_object:
            raise ValueError("Build schema before saving it.")
        
        out_path = f"{SCHEMAS_PATH}{self.dataset}/{self.db_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.schema_object, f, indent=4)
        
        print(f"✅ Schema saved to {out_path}")

    # load schema object from json
    def load_schema_json(self, repopulate_attributes=True):
        
        path = f"{SCHEMAS_PATH}{self.dataset}/{self.db_id}.json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"No schema file found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.schema_object = json.load(f)

        if repopulate_attributes:
            self.tables = list(self.schema_object.keys())
            #self.columns = {tbl: data.get("columns", []) for tbl, data in self.schema_object.items()}
            #self.primary_keys = {tbl: data.get("primary_keys", []) for tbl, data in self.schema_object.items()}
            #self.foreign_keys = {tbl: data.get("foreign_keys", []) for tbl, data in self.schema_object.items()}

        return self.schema_object
    
    def generate_schema_string(self):

        if not self.schema_object:
            raise RuntimeError("Schema object is not populated!")
        
        foreign_keys = []

        # db_id
        schema_string = f"## Database Name: {self.schema_object['db_id']} \n\n"

        # schema
        schema_string += "## Database Schema \n\n"

        # tables with columns
        for table_name, table_object in self.schema_object["schema"].items():

            schema_string += f"# Table: {table_name}\n[\n"
            
            for column_object in table_object["columns"]:
                schema_string += f"({column_object['name']}: {column_object['type'].upper()},"
                if column_object['pk'] == 1:
                    schema_string += " PRIMARY KEY,"
                if not column_object['notnull']:
                    schema_string += " NOT NULL"                
                schema_string += "),\n"
            schema_string += "]\n\n"

            for fk in table_object.get("foreign_keys", []):
                # foreign key column on this table
                fk_identifier = f"{table_name}.{fk['targetColumn']}"
                # PK it references:
                pk_identifier = f"{fk['sourceTable']}.{fk['sourceColumn']}"
                foreign_keys.append((fk_identifier, pk_identifier))
        
        # foreign keys
        if foreign_keys:
            schema_string += "## Foreign Keys \n"
            for fk_identifier, pk_identifier in foreign_keys:
                schema_string += f"{fk_identifier} REFERENCES {pk_identifier}\n"

        return schema_string


# utilities

def to_dict(cursor: sqlite3.Cursor):
    
    if cursor.description is None:
        return []

    # extract column names
    columns = [desc[0] for desc in cursor.description]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def normalize_type(declared_type: str) -> str:

    if not declared_type:
        return "BLOB"  # no type specified defaults to BLOB affinity
    
    t = declared_type.upper()
    if "INT" in t:
        return "INTEGER"
    elif any(x in t for x in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    elif "BLOB" in t:
        return "BLOB"
    elif any(x in t for x in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    elif any(x in t for x in ("DATE", "DATETIME", "TIME")):
        return "DATE"
    else:
        return "NUMERIC"

