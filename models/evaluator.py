
import os
import json
import numpy as np
from tqdm import tqdm
from func_timeout import func_timeout
from sentence_transformers import SentenceTransformer, util

from configs.paths import RESULTS_PATH, SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH
from external.testsuitesqleval.exec_eval import eval_exec_match
from external.bird.evaluation import execute_sql


EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en")

CLASS_WEIGHT = 0.3
RESPONSE_WEIGHT = 0.7

# message similarity thresholds
T_LOW = 0.40
T_HIGH = 0.80
IMPROPER_MAX_TOKENS = 20

TEMPLATES = {
    "ambiguous": [
        "Your question is ambiguous. Could you clarify what you mean?",
        "Do you mean one thing or another? Please specify so I can answer.",
        "Could you clarify what exactly you want me to retrieve?"
    ],
    "unanswerable": [
        "I cannot answer this because the information is not in the database schema.",
        "The data you are asking for is not stored in the provided tables.",
        "The database does not contain the information required to answer."
    ],
    "improper": [
        "You're welcome!",
        "Glad I could help.",
        "No problem, let me know if you need anything else."
    ],
}



class Evaluator:

    def __init__(self, dataset:str=None, model:str=None):

        self.dataset = dataset
        self.model = model
        
        if self.dataset == "spider":
            self.db_path = SPIDER_DATABASE_PATH
        elif self.dataset == "bird":
            self.db_path = BIRD_DATABASE_PATH

        with open(f"{RESULTS_PATH}{self.dataset}_{self.model}_results.json", "r") as f: 
            self.results = json.load(f)
        
        self.template_embeddings = {
            category: [EMBED_MODEL.encode(text, convert_to_tensor=True) for text in texts]
            for category, texts in TEMPLATES.items()
        }

        self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.model}_eval.json"


    def fit_sql(self):
        
        if os.path.exists(self.eval_path):
            raise Exception("Evaluation files already generated")

        total_score = 0

        for _, result in tqdm(enumerate(self.results)):

            classification_score = self.classification_accuracy(result)
            response_score = self.response_accuracy(result)
            fit_score = CLASS_WEIGHT * classification_score + RESPONSE_WEIGHT * response_score

            total_score += fit_score

            result["classification_score"] = classification_score
            result["response_score"] = response_score
            result["fit_score"] = fit_score
        
        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        print(f"FIT-SQL for {self.model} in {self.dataset}: {total_score / len(self.results)}")
        return total_score / len(self.results)


    def classification_accuracy(self, result_dict:dict):
        gold_type = result_dict.get("type_gold")
        pred_type = result_dict.get("response", {}).get("type")

        if gold_type is None: raise ValueError("Gold type must not be None.")
        if pred_type is None: return 0

        if normalize_type(gold_type) == normalize_type(pred_type):
            return 1
        else:
            return 0

    def response_accuracy(self, result_dict:dict):
        
        db_id = result_dict.get("db_id")
        gold_sql = result_dict.get("sql_gold")
        pred_sql = result_dict.get("response", {}).get("sql")
        gold_type = result_dict.get("type_gold")
        pred_type = result_dict.get("response", {}).get("type")
        message = result_dict.get("response", {}).get("message")

        if not gold_type or not db_id: raise ValueError("Gold type must not be None.")

        # llm system error (no type predicted)
        if pred_type is None or pred_type not in ["sql", "answerable", "improper", "unanswerable", "ambiguous"]:
            print("Invalid type prediction")
            # print(str(result_dict))
            return 0
        
        # not answerable but sql predicted
        empty_pred_sql = pred_sql is None or pred_sql == "null" or pred_sql == ""
        if normalize_type(gold_type) != "answerable" and not empty_pred_sql:
            return 0

        if normalize_type(gold_type) == "answerable":
            return self.execution_accuracy(db_id=db_id, gold_sql=gold_sql, pred_sql=pred_sql)
        else:
            return self.message_accuracy(message=message, templates=self.template_embeddings[normalize_type(gold_type)])

    def execution_accuracy(self, db_id:str, gold_sql:str, pred_sql:str):
        
        if pred_sql is None or pred_sql == "":
            return 0

        db = f"{self.db_path}{db_id}/{db_id}.sqlite"

        if self.dataset == "spider":
            exec_score = eval_exec_match(db=db, p_str=pred_sql, g_str=gold_sql, plug_value=False,
                                             keep_distinct=True, progress_bar_for_each_datapoint=False)
        elif self.dataset == "bird":
            try:
                exec_score = func_timeout(30, execute_sql, args=(pred_sql, gold_sql, db))
            except:
                exec_score = 0
        else:
            raise Exception("Uknown dataset during evaluation.")
        
        return exec_score

    def message_accuracy(self, message:str=None, templates:list=None):
        
        if message is None or message == "":
            return 0
        
        message_embedding = EMBED_MODEL.encode(message, convert_to_tensor=True)

        similarity_scores = []
        for template in templates:
            sim = util.cos_sim(message_embedding, template).item()
            similarity_scores.append(sim)

        s = max(similarity_scores)

        if s <= T_LOW:
            return 0
        elif s >= T_HIGH:
            return 1
        else:
            return (s - T_LOW) / (T_HIGH - T_LOW)

    def TDEX(self):
        
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception("Creat eval file with fit_sql() first.")
        
        for sample in tqdm(eval):

            if normalize_type(sample["type_gold"]) == "answerable":
                tdexa = sample["response_score"]
            else:
                tdexa = sample["classification_score"]
            
            sample["tdex_score"] = tdexa
        
        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(eval, f, indent=4)
        
        return True


    # fit-sql and tdex
    def analyze_fit_tdex(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception("Create eval file with fit_sql() first.")
        
        total_fit = 0
        total_tdex = 0
        for sample in eval:
            total_fit += sample["fit_score"]
            total_tdex += sample["tdex_score"]
        
        fit_score = round(total_fit / len(eval) * 100, 2)
        tdex_score = round(total_tdex / len(eval) * 100, 2)
        print(f"{self.dataset} | {self.model} | FIT: {fit_score} | TDEX: {tdex_score} | Length: {len(eval)}")
    
    # exa
    def analyze_exa(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception("Create eval file with fit_sql() first.")
        
        exa = 0
        exa_count = 0
        for sample in eval:
            if normalize_type(sample["type_gold"]) == "answerable":
                exa += sample["response_score"]
                exa_count += 1
        
        total_exa = round(exa / exa_count * 100, 2)
        print(f"{self.dataset} | {self.model} | ExA: {total_exa} | Length: {exa_count}")

    # classification and response accuracy
    def analyze_classification_response(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception("Create eval file with fit_sql() first.")
        
        total_class = 0
        total_resp = 0
        for sample in eval:
            total_class += sample["classification_score"]
            total_resp += sample["response_score"]
        
        class_score = round(total_class / len(eval) * 100, 2)
        resp_score = round(total_resp / len(eval) * 100, 2)
        print(f"{self.dataset} | {self.model} | Class: {class_score} | Resp: {resp_score} | Length: {len(eval)}")

    # fit-sql per type 
    def analyze_type_fit(self, datasets:list=None):

        categories = ["answerable", "unanswerable", "ambiguous", "improper"]
        totals = {key: {"fit": 0.0, "count": 0} for key in categories}

        for dataset in datasets:
            # eval file needs to be created first
            eval_path = f"{RESULTS_PATH}{dataset}_{self.model}_eval.json" # use passed datasets
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f: 
                    eval = json.load(f)
            else:
                raise Exception("Create eval file with fit_sql() first.")
            
            for sample in eval:
                t = normalize_type(sample["type_gold"])
                totals[t]["fit"] += sample["fit_score"]
                totals[t]["count"] += 1
            
        averages = {
            key: (totals[key]["fit"] / totals[key]["count"] * 100
                if totals[key]["count"] > 0 else None)
            for key in categories
        }
        
        print(f"{self.model} | Class accuracies:")
        for cls, avg in averages.items():
            print(f"  {cls:12s} : {avg:.3f}" if avg is not None else f"  {cls:12s} : None")

    def analyze_amb_type(self, datasets:list=None):
        categories = ["ambiguous_column", "ambiguous_temporal", "ambiguous_aggregation", 
                      "ambiguous_schema", "ambiguous_linguistic"]
        totals = {key: {"fit": 0.0, "count": 0} for key in categories}

        for dataset in datasets:
            # eval file needs to be created first
            eval_path = f"{RESULTS_PATH}{dataset}_{self.model}_eval.json" # use passed datasets
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f: 
                    eval = json.load(f)
            else:
                raise Exception("Create eval file with fit_sql() first.")
            
            for sample in eval:
                if normalize_type(sample["type_gold"]) == "ambiguous":
                    t = sample["type_gold"]
                    totals[t]["fit"] += sample["fit_score"]
                    totals[t]["count"] += 1
            
        averages = {
            key: (totals[key]["fit"] / totals[key]["count"] * 100
                if totals[key]["count"] > 0 else None)
            for key in categories
        }
        
        print(f"{self.model} | Ambiguous class accuracies:")
        for cls, avg in averages.items():
            print(f"  {cls:12s} : {avg:.3f}" if avg is not None else f"  {cls:12s} : None")

    def analyze_confusion(self, models:list=None):

        gold_classes = ["answerable", "unanswerable", "ambiguous", "improper"]
        pred_classes = ["answerable", "unanswerable", "ambiguous", "improper", None]

        # Initialize empty confusion matrix
        matrix = {
            gold: {pred: 0 for pred in pred_classes}
            for gold in gold_classes
        }    
        
        for model in models:
            # eval file needs to be created first
            eval_path = f"{RESULTS_PATH}{self.dataset}_{model}_eval.json" # use passed models
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f: 
                    eval = json.load(f)
            else:
                raise Exception("Create eval file with fit_sql() first.")        
        
            for sample in eval:
                gold = normalize_type(sample.get("type_gold"))
                pred = normalize_type(sample.get("response", {}).get("type"))

                if gold not in gold_classes:
                    continue
                if pred not in pred_classes:
                    pred = None

                matrix[gold][pred] += 1
        
        
        pred_classes = list(next(iter(matrix.values())).keys())

        print(self.dataset)
        header = "Gold \\ Pred".ljust(14) + " ".join([str(c).ljust(12) for c in pred_classes])
        print(header)
        print("-" * len(header))

        for gold, row_data in matrix.items():
            row = gold.ljust(14)
            for pred in pred_classes:
                row += str(row_data[pred]).ljust(12)
            print(row)
        
    def analyze_errors(self, datasets:list=None, models:list=None):
        
        total_count = 0
        type_error_count = 0 # wrong type prediction
        sql_produced_count = 0 # SQL produced when not allowed
        clarification_count = 0
        explanation_count = 0
        invalid_sql_count = 0
        output_formatting_errors = 0

        for dataset in datasets:
            for model in models:

                eval_path = f"{RESULTS_PATH}{dataset}_{model}_eval.json" # use passed models
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f: 
                        eval = json.load(f)
                else:
                    raise Exception("Create eval file with fit_sql() first.")   

                for sample in eval:
                    total_count += 1
                    
                    if sample["classification_score"] != 1:
                        type_error_count += 1
                    
                    sql = sample.get("response", {}).get("sql")
                    sql_empty = sql is None or sql == "" or sql == "null"
                    if normalize_type(sample["type_gold"]) != "answerable" and not sql_empty:
                        sql_produced_count += 1
                    
                    if normalize_type(sample["type_gold"]) != "answerable":
                        pass



def normalize_type(type_name:str):

    if type_name is None:
        return None

    if "ambiguous" in type_name.lower():
        return "ambiguous"
    elif "improper" in type_name.lower():
        return "improper"
    elif "unanswerable" in type_name.lower():
        return "unanswerable"
    elif "answerable" in type_name.lower() or "sql" in type_name.lower():
        return "answerable"
    else: 
        try:
            return type_name.lower()
        except:
            return type_name