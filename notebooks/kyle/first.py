from model.heuristic.mobile.extractor import *
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

E = Extractor()
for index, row in mobile_df.iterrows():
    if index == 609:
        break

    extracted = E.extract_from_title(row["title"])
    print(row["title"])
    print(json.dumps(extracted, indent=4))
