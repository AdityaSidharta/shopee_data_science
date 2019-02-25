from model.heuristic import mobile
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

M = mobile.Predictor()
for index, row in mobile_df.iterrows():
    if index == 2:
        break

    prediction = M.predict(row["title"])
    print(row["title"])
    print(json.dumps(prediction, indent=4))
