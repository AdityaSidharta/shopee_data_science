from utils.fonAPI import FonApi
import utils.envs as env
import json
import collections
import os
import re


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
            pass
        elif len(self.phones_from_api) == 1:
            # print(json.dumps(self.phones_from_api[0]["technology"], indent=4))
            pass
        extracted["Operating System"] = self.get_os()
        extracted["Network Connections"] = self.get_networks()
        capacity = extracted["Storage Capacity"]
        (capacity, memory) = self.get_cap_n_mem(capacity)
        extracted["Storage Capacity"] = capacity
        extracted["Memory RAM"] = memory
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

    def get_cap_n_mem(self, capacity):
        given_mems = [ # from profile_*.json
                "4gb", "2gb", "1.5gb", "16gb", "512mb",
                "8gb", "3gb", "10gb", "1gb", "6gb"
                ]
        given_caps = [
                "256gb", "1.5gb", "128gb", "512mb", "64gb",
                "512gb", "8gb", "4mb", "6gb", "4gb", "2gb",
                "128mb", "32gb", "256mb", "10gb", "3gb",
                "1gb", "16gb"
                ]
        options = []
        memory = ""
        phones = self.phones_from_api
        # TODO be more flexible
        if len(phones) != 1 or "internal" not in phones[0]:
            return capacity, memory
        internal = phones[0]["internal"].replace("3 RAM", "3 GB RAM").replace("4/6 GB RAM", "4 GB RAM").replace("64/32 GB, 4 GB RAM", "64 GB, 4 GB RAM")
        re_matches = re.finditer(r"((?P<cap>\d*\ [A-Z]{2}),\ (?P<mem>\d*\ [A-Z]{2}))\ RAM", internal, re.MULTILINE)
        # Transform text
        for match in re_matches:
            option = match.groupdict()
            for key in option:
                option[key] = option[key].lower().replace(" ", "")
            options.append(option)
        if len(options) == 0:
            return capacity, memory
        elif len(options) == 1:
            memory = options[0]["mem"]
            if memory not in given_mems: # if not in profile*.json
                return capacity, ""
            else:
                if capacity == "" and options[0]["cap"] in given_caps:
                    capacity = options[0]["cap"]
                return capacity, memory
        else:
            options_dict = {}
            for option in options:
                options_dict[option["cap"]] = option["mem"]
            if capacity in options_dict:
                return capacity, options_dict[capacity]
            elif capacity == "":
                return capacity, memory
        return capacity, memory

    def string_found(self, substr, mainstr):
        substr = " " + substr.strip() + " "
        mainstr = " " + mainstr.strip() + " "
        return substr in mainstr
