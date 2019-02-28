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
    all_models.append((brand, device))
all_models.sort(key=lambda x: len(x[1]), reverse=True)

# Scrape sure things from API
for i, device in enumerate(all_models):
    print(device)
    En.enrich({
        "Phone Model": device[1],
        "Brand": device[0]
        })

# # Clean up the missing ones
# hardcode_dir = "./data/fono_api"
# filenames = [i for i in os.walk(hardcode_dir)][0][-1]
# for filename in filenames:
#     phones = json.loads(open(hardcode_dir + "/" + filename).read())
#     if len(phones) > 0:
#         continue
#     (brand, phone_model) = filename[:-5].split("_")
#     device = (brand, phone_model)
#     print(device)
#     break
# 
# En.enrich({
#     "Phone Model": "aspire 6",
#     "Brand": "acer"
#     })
