import os
from tqdm import tqdm

from models.schema_builder import SchemaBuilder
from configs.paths import SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH

"""

    creates schema representation in json via
    SchemaBuilder and stores files in configs.paths.SCHEMA_PATHS

"""

SPIDER_DATASETS = os.listdir(SPIDER_DATABASE_PATH)
BIRD_DATASETS = os.listdir(BIRD_DATABASE_PATH)

print("Starting schema generation for spider.")
for db in tqdm(SPIDER_DATASETS):
    with SchemaBuilder(dataset="spider", db_id=db) as sb:
        sb.build_schema_object()
        sb.save_schema_json()

print("Starting schema generation for bird.")
for db in tqdm(BIRD_DATASETS):
    with SchemaBuilder(dataset="bird", db_id=db) as sb:
        sb.build_schema_object()
        sb.save_schema_json()


