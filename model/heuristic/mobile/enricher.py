from utils.fonAPI import FonApi
import utils.envs as env
import json
import collections
import os


class Enricher:
    """Enriches given mobile data."""

    def __init__(self):
        self.Fon = FonApi(env.fono_key)

    def enrich(self, extracted):
        phones = self.search(extracted)
        if len(phones) == 0:
            print("No Matches Found.")
        for phone in phones:
            print(json.dumps(phone, indent=4))
        return extracted

    def search(self, extracted):
        phones = []
        brand = extracted["Brand"]
        device = extracted["Phone Model"]
        if device:
            phones = self.get_phones(device, brand)
        else:
            phones = []
        return phones

    def get_phones(self, device, brand):
        """Get phones from file cache or API."""
        fono_path = env.fono_data_path
        filename = "{}_{}.json".format(brand, device)
        filepath = os.path.join(fono_path, filename)
        if os.path.exists(filepath):
            phones = json.loads(open(filepath, "r").read())
        else:
            if brand and device:
                phones = self.Fon.getdevice(device, brand=brand)
            else:
                phones = self.Fon.getdevice(device)
            if phones == "No Matching Results Found.":
                if device.split()[-1] in [str(i) for i in range(1, 10)]:
                    # Recursively get phones if last word
                    # is a number from 1-9 (e.g. Zenfone zc 1)
                    self.get_phones(device[:-2], brand)
                phones = []
            open(filepath, "w").write(json.dumps(phones))
        return phones
