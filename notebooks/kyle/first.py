from model.heuristic.mobile.extractor import *
from model.heuristic.mobile.enricher import *
import utils.envs as env

import pandas as pd
import numpy as np
import json
import os


mobile_df = pd.read_csv(
        env.mobile_train_repo
        ).fillna('')
mobile_profile = json.loads(
        open(env.mobile_profile_json, "r").read()
        )

Ex = Extractor()
En = Enricher()
for index, row in mobile_df.iterrows():
    if index == 2:
        break

    extracted = Ex.extract_from_title(row["title"])
    enriched = En.enrich(extracted)
    print(row["title"])
    print(json.dumps(enriched, indent=4))
