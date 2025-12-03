import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from models.prompt import Prompter
from models.schema_builder import SchemaBuilder
from configs.paths import QUESTIONS_PATH, RESULTS_PATH

load_dotenv()

# DATASET = "spider" # spider | bird
# MODEL = "gemini-2.5-pro"
MODELS = {
    "gpt-5": {"provider": "openai", "model": "gpt-5"},
    # "gemini-3": {"provider": "google", "model": "gemini-3-pro-preview"},
    "gemini-2.5-pro": {"provider": "google", "model": "gemini-2.5-pro"},
    "qwen-3-80B": {"provider": "together", "model": "Qwen/Qwen3-Next-80B-A3B-Thinking"},
    # "deepseek-3.1": {"provider": "together", "model": "deepseek-ai/DeepSeek-V3.1"},
    "llama-3.3-70B": {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    parser.add_argument("--model", type=str, choices=["gpt-5", "gemini-2.5-pro", "qwen-3-80B", "llama-3.3-70B"], default="gpt-5")
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL = args.model

    # load questions (only answerable)
    with open(f"{QUESTIONS_PATH}questions_{DATASET}.json", "r") as f: 
        samples = json.load(f)

    schema_strings = {}

    responses = []

    # json as main results file
    json_path = f"{RESULTS_PATH}{DATASET}_{MODEL}_results.json"
    if os.path.exists(json_path):
        raise Exception("Responses already generated.")

    # jsonl as backup
    jsonl_path = f"{RESULTS_PATH}{DATASET}_{MODEL}_results.jsonl"
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                responses.append(json.loads(line))

    # with open(jsonl_path, "w", encoding="utf-8"): pass # create new empty jsonl backup file
    jsonl_out = open(jsonl_path, "a", encoding="utf-8")

    if len(responses) == len(samples):
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4)
        raise Exception("Responses already generated.")

    start_index = len(responses)
    print(f"Starting generating responses at index {start_index}")

    for i, sample in tqdm(enumerate(samples[start_index:], start=start_index)):

        db_id = sample["db_id"]

        if db_id not in schema_strings:
            sb = SchemaBuilder(dataset=DATASET, db_id=db_id)
            sb.load_schema_json(repopulate_attributes=True)
            schema_strings[db_id] = sb.generate_schema_string()
        
        p = Prompter(
            provider=MODELS[MODEL]["provider"], model=MODELS[MODEL]["model"], schema_string=schema_strings[db_id]
        )

        # print(f"Generating response {i}")
        response = p.ask_question(question=sample["question"]) # returns llm response dictionary

        response["type_gold"] = sample["type"]
        response["sql_gold"] = sample["sql"]
        response["db_id"] = db_id
        response["index"] = i

        responses.append(response)

        jsonl_out.write(json.dumps(response) + "\n")
        jsonl_out.flush()

    jsonl_out.close()

    # create final json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    print(f"✅ Results of {DATASET} saved to {json_path}")


