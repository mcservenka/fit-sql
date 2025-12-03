import os
import json
import random
from collections import Counter

from models.ambiguity_detector import AmbiguityDetector
from configs.paths import (
    SPIDER_DEV_PATH, SPIDER_DEV_AUG_PATH, BIRD_DEV_PATH, BIRD_DEV_AUG_PATH,
    TEMP_IMPROPER, SPIDER_TRAIN_PATH, TEMP_AMB_LINGUISTIC, TEMP_AMB_COLUMN,
    TEMP_AMB_AGGREGATION, TEMP_AMB_SCHEMA, TEMP_AMB_TEMPORAL, SCHEMAS_PATH,
    QUESTIONS_PATH
)


PLACEHOLDER = "XPLACEHOLDERX" # placeholder used in templates


class QuestionGenerator:

    def __init__(self, dataset:str="spider"):

        self.dataset = dataset

        if dataset == "spider":
            self.dev_path = SPIDER_DEV_PATH
            self.aug_path = SPIDER_DEV_AUG_PATH
        elif dataset == "bird":
            self.dev_path = BIRD_DEV_PATH
            self.aug_path = BIRD_DEV_AUG_PATH

        # load questions (only answerable)
        with open(self.dev_path, "r") as f: 
            data = json.load(f)

        self.data = [
            {
                "db_id": item.get("db_id"),
                "question": item.get("question"),
                "sql": item.get("SQL") or item.get("query"),
                "type": "answerable"
            }
            for item in data
        ]

        self.distinct_dbs = {item["db_id"] for item in self.data}
        self.db_schemas = {} # holds all schema representations with table name as key
        self.augmentation_statistics = {} # tracks how many augmented samples were added for each database

        # contain templates (only loaded if specific method is called)
        self.improper_templates = []
        self.column_ambiguity_templates = []
        self.aggregation_ambiguity_templates = []
        self.linguistic_ambiguity_templates = []
        self.schema_ambiguity_templates = []
        self.temporal_ambiguity_templates = []
        
        self.train_set = [] # contains training set (only loaded if training set is included in generate_unanswerable)

        # if questions already exist, load them, so they can be used for statistics
        if os.path.exists(f"{QUESTIONS_PATH}questions_{self.dataset}.json"):
            with open(f"{QUESTIONS_PATH}questions_{self.dataset}.json", "r") as f: 
                self.data = json.load(f)

    
    def generate_improper(self, db_id:str=None, n:int=10):

        # load improper templates
        if not self.improper_templates:
            with open(TEMP_IMPROPER, "r") as f: 
                self.improper_templates = json.load(f)

        samples = random.sample(self.improper_templates, n)

        for sample in samples:
            self.data.append({
                "db_id": db_id, 
                "question": sample,
                "sql": None,
                "type": "improper"
            })
        
        return samples

    def generate_unanswerable(self, db_id:str=None, n:int=20, include_train=True):
        
        data = self.data

        # load train set (only for spider)
        if self.dataset == "spider" and include_train and not self.train_set:
            with open(SPIDER_TRAIN_PATH, "r") as f: 
                self.train_set = json.load(f)
                data = self.data + self.train_set

        # filter data for allowed samples
        excluded = EXCLUDED_DATABASES[self.dataset].get(db_id)
        excluded.append(db_id)

        filtered_data = [
            item for item in data
            if not item.get("type") or item.get("type") == "answerable"
            and item["db_id"] not in excluded
        ]

        samples = random.sample(filtered_data, n)

        for sample in samples:
            self.data.append({
                "db_id": db_id, 
                "question": sample.get("question"),
                "sql": None,
                "type": "unanswerable"
            })
        
        return samples
        
    def generate_ambiguous(self, db_id:str=None, n:int=5,
                           column:bool=True,
                           aggregation:bool=True,
                           linguistic:bool=True,
                           schema:bool=True,
                           temporal:bool=True):
        
        # load templates if necessary
        self._load_ambiguity_templates()

        # load schema representation
        if not hasattr(self.db_schemas, db_id):
            with open(f"{SCHEMAS_PATH}{self.dataset}/{db_id}.json", "r") as f: 
                self.db_schemas[db_id] = json.load(f)

        # detect ambiguity options
        ag = AmbiguityDetector(self.db_schemas[db_id])

        if column:
            ag.detect_column_ambiguity()
            
            for col in ag.ambiguous_columns:
                
                coltype = col.get("types")[0]
                if len(col.get("types")) > 1 or coltype in ["BLOB", "TEXT"]:
                    samples = random.sample(self.column_ambiguity_templates["nominal"], n)
                elif coltype in ["REAL", "NUMERIC", "INTEGER"]:
                    samples = random.sample(self.column_ambiguity_templates["numeric"], n)
                elif coltype in ["DATE", "DATETIME"]:
                    samples = random.sample(self.column_ambiguity_templates["temporal"], n)
                else:
                    raise ValueError(f"Type of Column not found: {col}")

                for sample in samples:
                    if col.get("column_name"):
                        self.data.append({
                            "db_id": db_id, 
                            "question": sample.replace(PLACEHOLDER, col.get("column_name")),
                            "sql": None,
                            "type": "ambiguous_column"
                        })
                    else:
                        raise ValueError(f"Column name not found in {col}")

        if temporal:
            ag.detect_temporal_ambiguity()

            temp_amb_questions = self._generate_ambiguity_questions(
                templates=self.temporal_ambiguity_templates,
                detection_result=ag.ambiguous_temporal_tables,
                n=n
            )

            for question in temp_amb_questions:
                self.data.append({
                    "db_id": db_id, 
                    "question": question,
                    "sql": None,
                    "type": "ambiguous_temporal"
                })
        
        if aggregation:
            ag.detect_aggregation_ambiguity()

            agg_amb_questions = self._generate_ambiguity_questions(
                templates=self.aggregation_ambiguity_templates,
                detection_result=ag.ambiguous_aggregation_tables,
                n=n
            )

            for question in agg_amb_questions:
                self.data.append({
                    "db_id": db_id, 
                    "question": question,
                    "sql": None,
                    "type": "ambiguous_aggregation"
                })
        
        if schema:
            ag.detect_key_ambiguity()

            schema_amb_questions = self._generate_ambiguity_questions(
                templates=self.schema_ambiguity_templates,
                detection_result=ag.ambiguous_schema_graph_tables,
                n=n
            )

            for question in schema_amb_questions:
                self.data.append({
                    "db_id": db_id, 
                    "question": question,
                    "sql": None,
                    "type": "ambiguous_schema"
                })

        if linguistic:
            samples = random.sample(self.linguistic_ambiguity_templates, n)
            tables = list(self.db_schemas[db_id]["schema"].keys())

            for sample in samples:
                question = sample.replace(PLACEHOLDER, random.choice(tables))
                self.data.append({
                    "db_id": db_id, 
                    "question": question,
                    "sql": None,
                    "type": "ambiguous_linguistic"
                })

    
    def calculate_statistics(self):
        c = Counter(d["type"] for d in self.data)
        total = sum(c.values())

        print("\n" + "=" * 30)
        print("        DATA STATISTICS")
        print(f"        {self.dataset}")
        print("=" * 30)

        for t, count in c.items():
            print(f"{t:<15} : {count}")

        print("-" * 30)
        print(f"Total samples     : {total}")
        print("=" * 30 + "\n")

    def save_questions_json(self):
        if not self.data:
            raise ValueError("Generate questions before saving them.")
        
        out_path = f"{QUESTIONS_PATH}questions_{self.dataset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
        
        print(f"✅ Questions of {self.dataset} saved to {out_path}")

    
    # utils

    def _generate_ambiguity_questions(self, templates:list, detection_result:list, n:int):
        amb_questions = []
        for tbl in detection_result:
            samples = random.sample(templates, n)
            for sample in samples:
                if tbl.get("table"):
                    amb_questions.append(sample.replace(PLACEHOLDER, tbl.get("table")))
                else:
                    raise ValueError(f"Table not defined in {tbl}")
        
        return amb_questions

    def _load_ambiguity_templates(self):
        
        # column ambiguity
        if not self.column_ambiguity_templates:
            with open(TEMP_AMB_COLUMN, "r") as f: 
                self.column_ambiguity_templates = json.load(f)
                print(f"Column Ambiguity (Nominal) Length: {len(self.column_ambiguity_templates['nominal'])}")
                print(f"Column Ambiguity (Numeric) Length: {len(self.column_ambiguity_templates['numeric'])}")
                print(f"Column Ambiguity (Temporal) Length: {len(self.column_ambiguity_templates['temporal'])}")

        # aggregation ambiguity
        if not self.aggregation_ambiguity_templates:
            with open(TEMP_AMB_AGGREGATION, "r") as f: 
                self.aggregation_ambiguity_templates = json.load(f)
                print(f"Aggregation Ambiguity Length: {len(self.aggregation_ambiguity_templates)}")

        # linguistic ambiguity
        if not self.linguistic_ambiguity_templates:
            with open(TEMP_AMB_LINGUISTIC, "r") as f: 
                self.linguistic_ambiguity_templates = json.load(f)
                print(f"Linguistic Ambiguity Length: {len(self.linguistic_ambiguity_templates)}")
        
        # schema ambiguity
        if not self.schema_ambiguity_templates:
            with open(TEMP_AMB_SCHEMA, "r") as f: 
                self.schema_ambiguity_templates = json.load(f)
                print(f"Schema Ambiguity Length: {len(self.schema_ambiguity_templates)}")
                
        # temporal ambiguity
        if not self.temporal_ambiguity_templates:
            with open(TEMP_AMB_TEMPORAL, "r") as f: 
                self.temporal_ambiguity_templates = json.load(f)
                print(f"Temporal Ambiguity Length: {len(self.temporal_ambiguity_templates)}")

        # improper templates
        with open(TEMP_IMPROPER, "r") as f:
            self.improper_templates = json.load(f)
            print(f"Improper Length: {len(self.improper_templates)}")
    
        

# databases which are excluded when sampling unanswerable questions
EXCLUDED_DATABASES = {
    "spider": {
        'battle_death': [""],
        'car_1': [""],
        'concert_singer': ["singer"],
        'course_teach': [""],
        'cre_Doc_Template_Mgt': [""],
        'dog_kennels': ["pets_1"],
        'employee_hire_evaluation': [""],
        'flight_2': [""],
        'museum_visit': [""],
        'network_1': [""],
        'orchestra': [""],
        'pets_1': ["dog_kennels"],
        'poker_player': [""],
        'real_estate_properties': [""],
        'singer': ["concert_singer"],
        'student_transcripts_tracking': [""],
        'tvshow': [""],
        'voter_1': [""],
        'world_1': [""],
        'wta_1': [""]
    },
    "bird": {
        'california_schools': ["student_club"],
        'card_games': [""],
        'codebase_community': [""],
        'debit_card_specializing': [""],
        'european_football_2': [""],
        'financial': [""],
        'formula_1': [""],
        'student_club': ["california_schools"],
        'superhero': [""],
        'thrombosis_prediction': [""],
        'toxicology': [""]
    }
}

        

