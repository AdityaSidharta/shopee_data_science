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


print(mobile_df["title"].head())
print(json.dumps(mobile_profile, indent=4))
for index, row in mobile_df.iterrows():
    if index == 1:
        break
    print(index, row["title"], row["Operating System"])

print(mobile.predict())
