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

En = Enricher()
# Ex = Extractor()
# for index, row in mobile_df.iterrows():
#     if index == 1:
#         break
# 
#     extracted = Ex.extract_from_title(row["title"])
#     enriched = En.enrich(extracted)
#     print(row["title"])
#     print(json.dumps(enriched, indent=4))


# Get phone models + brand names
all_models = []
for phone_model in mobile_profile["Phone Model"]:
    brand = phone_model.split()[0]
    device = " ".join(phone_model.split()[1:])
    if brand in mobile_profile["Brand"]:
        all_models.append((brand, device))
all_models.sort(key=lambda x: len(x[1]), reverse=True)

# Scrape sure things from API
for i, device in enumerate(all_models):
    print(device)
    En.enrich({
        "Phone Model": device[1],
        "Brand": device[0]
        })
    # print(device)
    # # if i > 40 and i < 50:
    # #     print(device)
    # #     En.enrich({
    # #         "Phone Model": device[1],
    # #         "Brand": device[0]
    # #         })
    # if i == 50:
    #     break
