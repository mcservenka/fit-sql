import argparse

from models.evaluator import Evaluator

# DATASET = ["spider", "bird"] # spider | bird
# MODEL = ["gpt-5", "gemini-2.5-pro", "qwen-3-80B", "llama-3.3-70B"] #"gpt-5" # gpt-5 | gemini-2.5-pro | qwen-3-80B | llama-3.3-70B

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    parser.add_argument("--model", type=str, choices=["gpt-5", "gemini-2.5-pro", "qwen-3-80B", "llama-3.3-70B"], default="gpt-5")
    args = parser.parse_args()

    
    ev = Evaluator(dataset=args.dataset, model=args.model)
    ev.fit_sql()