import json
from configs.paths import SPIDER_DEV_PATH, BIRD_DEV_PATH

def get_dev_dbs(dataset:str = "spider"):

    if dataset == "spider":
        p = SPIDER_DEV_PATH
    elif dataset == "bird":
        p = BIRD_DEV_PATH

    with open(p, "r") as f:
        dev = json.load(f)
    
    db_ids = list()
    for d in dev:
        db_ids.append(d["db_id"])

    return set(db_ids)