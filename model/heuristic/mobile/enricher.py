from utils.fonAPI import FonApi
import utils.envs as env
import json
import collections
import os


class Enricher:
    """Enriches given mobile data."""

    def __init__(self):
        self.Fon = FonApi(env.fono_key)
        self.phones_from_api = []

    def enrich(self, extracted):
        self.phones_from_api = self.search(extracted)
        extracted["Enriched"] = True
        if len(self.phones_from_api) == 0:
            extracted["Enriched"] = False
        elif len(self.phones_from_api) > 1:
            # print(json.dumps(self.phones_from_api, indent=4))
            extracted["Operating System"] = self.get_os()
        elif len(self.phones_from_api) == 1:
            # print(json.dumps(self.phones_from_api[0]["technology"], indent=4))
            extracted["Operating System"] = self.get_os()
            extracted["Network Connections"] = self.get_networks()
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
        # Very messy. Needs rewrite.
        fono_path = env.fono_data_path
        filename = "{}_{}.json".format(brand, device)
        filepath = os.path.join(fono_path, filename)
        if os.path.exists(filepath):
            phones = json.loads(open(filepath, "r").read())
        else:
            # For now
            return []
            if brand and device:
                phones = self.Fon.getdevice(device, brand=brand)
            else:
                phones = self.Fon.getdevice(device)
            if phones == "No Matching Results Found.":
                if device.split()[-1] in [str(i) for i in range(1, 10)] and len(device) > 1:
                    # Recursively get phones if last word
                    # is a number from 1-9 (e.g. Zenfone zc 1)
                    self.get_phones(device[:-2], brand)
                phones = []
            open(filepath, "w").write(json.dumps(phones))
        return phones

    def get_os(self):
        phones = self.phones_from_api
        os_set = set()
        known_systems = [
                "ios", "android", "symbian", "samsung os",
                "blackberry os", "windows", "nokia os"
                ]
        excluded_systems = [
                "tizen-based", "tizen", "watchos"
                ]
        for phone in phones:
            if "os" not in phone:
                return ""
            os_str = phone["os"].lower()
            for os in (known_systems + excluded_systems):
                if self.string_found(os, os_str):
                    os_set.add(os)
        if len(list(os_set)) == 1:
            os = list(os_set)[0]
            if os in excluded_systems:
                return ""
            else:
                return os
        elif len(list(os_set)) > 1:
            # TODO account for more than one OS
            return ""
        else: # 0
            return ""

    def get_networks(self):
        phones = self.phones_from_api
        networks = set()
        for phone in phones:
            if "technology" not in phone:
                continue
            technology = phone["technology"].lower()
            if self.string_found("gsm", technology):
                networks.add("4g")
            elif self.string_found("hspa", technology):
                networks.add("3.5g")
            elif self.string_found("cdma", technology):
                networks.add("3g")
            elif self.string_found("evdo", technology):
                networks.add("2g")
            elif self.string_found("no cellular", technology):
                networks.add("none")
        if len(list(networks)) == 1:
            network = list(networks)[0]
            if network == "none":
                return ""
            else:
                return network
        elif len(list(networks)) > 1:
            # TODO account for more than one OS
            return ""
        else: # 0
            return ""

    def string_found(self, substr, mainstr):
        substr = " " + substr.strip() + " "
        mainstr = " " + mainstr.strip() + " "
        return substr in mainstr
