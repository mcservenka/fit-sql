#import torch
#from sentence_transformers import SentenceTransformer, util

# EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class AmbiguityDetector:

    def __init__(self, schema_object: dict):
        
        if not schema_object.get("dataset") or not schema_object.get("db_id"):
            raise Warning("Invalid schema object.")
        
        self.schema = schema_object.get("schema")
        self.dataset = schema_object.get("dataset")
        self.db_id = schema_object.get("db_id")

        self.ambiguous_columns = None
        self.ambiguous_aggregation_tables = None
        self.ambiguous_temporal_tables = None
        self.ambiguous_fk_tables = None


    def detect_column_ambiguity(self, ignore_ids=True):
        
        # ignore_ids: if true columns that end with "id" are ignored as well
        # output format: { "column_name": "population", "tables": ["city", "country"] }
        
        # create object with all column names as key and tables array in which they appear as value
        column_index = {}
        for table, meta in self.schema.items():
            for col in meta.get("columns", []):
                col_name = col["name"].lower()
                col_type = (col.get("typegroup") or "").upper()
                if col_name not in column_index:
                    column_index[col_name] = {"tables": [], "types": []}
                column_index[col_name]["tables"].append(table)
                if col_type and col_type not in column_index[col_name]["types"]:
                    column_index[col_name]["types"].append(col_type)

        # filter for columns that occur in more than one table
        duplicate_columns = {
            name: info
            for name, info in column_index.items()
            if len(info["tables"]) > 1
        }
        
        fk_links = set()
        fk_tables = [(tbl, meta.get("foreign_keys", [])) for tbl, meta in self.schema.items()]

        # get cases where tables are linked by foreign keys
        for t, fks in fk_tables:
            for fk in fks:
                pair = tuple(sorted([t, fk["sourceTable"]]))
                fk_links.add(pair)

        ambiguous_columns = []
        
        # only extract unlinked columns
        for col_name, info in duplicate_columns.items():

            if ignore_ids and col_name.lower().endswith("id"):
                continue
            
            tables = info["tables"]
            unlinked_pairs = []
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    pair = tuple(sorted([tables[i], tables[j]]))
                    if pair not in fk_links:
                        unlinked_pairs.append(pair)

            if unlinked_pairs:
                ambiguous_columns.append({
                    "column_name": col_name,
                    "tables": list({tbl for pair in unlinked_pairs for tbl in pair}),
                    "types": info["types"]
                })

        self.ambiguous_columns = ambiguous_columns
        return ambiguous_columns


    def detect_semantical_ambiguity(self, threshold=0.8):
        
        # get all columns of the database
        columns = []
        for table, meta in self.schema.items():
            for col in meta.get("columns", []):
                columns.append({"table": table, 
                                "column": col["name"], 
                                "type": col.get("typegroup", "")})
        
        col_names = [c["column"] for c in columns]

        embeddings = self._compute_embeddings(col_names)

        pairs = self._find_similar_pairs(col_names, embeddings, threshold)

        print(pairs)


    def detect_aggregation_ambiguity(self):

        numeric_groups = {"INTEGER", "REAL", "NUMERIC"}
        ambiguous_aggregation_tables = []

        for table, meta in self.schema.items():
            numeric_cols = [
                col for col in meta.get("columns", [])
                if (col.get("typegroup") or "").upper() in numeric_groups
            ]

            # only ambiguous if more than one numeric column
            if len(numeric_cols) > 1:
                ambiguous_aggregation_tables.append({
                    "table": table,
                    "numeric_columns": [c["name"] for c in numeric_cols],
                    "typegroups": list({c["typegroup"] for c in numeric_cols})
                })

        self.ambiguous_aggregation_tables = ambiguous_aggregation_tables
        return ambiguous_aggregation_tables


    def detect_temporal_ambiguity(self):

        temporal_groups = {"DATETIME", "DATE", "TIME"}
        ambiguous_temporal_tables = []

        for table, meta in self.schema.items():
            temporal_cols = [
                col for col in meta.get("columns", [])
                if (col.get("typegroup") or "").upper() in temporal_groups
            ]

            if len(temporal_cols) > 0:
                ambiguous_temporal_tables.append({
                    "table": table,
                    "temporal_columns": [c["name"] for c in temporal_cols],
                    "typegroups": list({c["typegroup"] for c in temporal_cols})
                })

        self.ambiguous_temporal_tables = ambiguous_temporal_tables
        return ambiguous_temporal_tables
    

    def detect_key_ambiguity(self):
        
        ambiguous_fk_tables = []

        for table, meta in self.schema.items():
            fks = meta.get("foreign_keys", [])
            if not fks or len(fks) < 2:
                continue  # skip tables with fewer than 2 foreign keys

            # extract distinct referenced tables
            referenced_tables = {fk.get("sourceTable") for fk in fks if fk}
            if len(referenced_tables) >= 2: # only if at least two different tables are referenced
                ambiguous_fk_tables.append({ "table": table, "foreign_keys": fks })

        self.ambiguous_schema_graph_tables = ambiguous_fk_tables
        return ambiguous_fk_tables
    
    
    # utils
    #def _compute_embeddings(self, texts):
    #    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    #    return EMBEDDING_MODEL.encode(
    #        texts,
    #        convert_to_tensor=True,
    #        normalize_embeddings=True  # important: cosine = dot product
    #    )
    
    def _find_similar_pairs(self, names, embeddings, threshold=0.8):
        """
        names: list of strings (column names)
        embeddings: normalized tensor embeddings
        threshold: cosine similarity threshold
        """
        sim_matrix = None#torch.matmul(embeddings, embeddings.T)
        n = len(names)

        pairs = []

        for i in range(n):
            for j in range(i + 1, n):  # ensures no duplicates + no self-pairs
                score = sim_matrix[i, j].item()
                if score >= threshold and score < 0.99:
                    pairs.append({
                        "col1": names[i],
                        "col2": names[j],
                        "similarity": score
                    })
        return pairs


