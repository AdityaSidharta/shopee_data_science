from utils.fonAPI import FonApi
import utils.envs as env
import json
import collections
import os
import re


class Enricher:
    """Enriches given mobile data."""

    def __init__(self):
        with open(env.gsm_arena, "r") as f:
            self.gsmarena = json.loads(f.read())
        with open(env.not_trained_devices, "r") as f:
            self.not_trained_devices = json.loads(f.read())
        self.Fon = FonApi(env.fono_key)
        self.phones_from_api = []

    def enrich(self, extracted):
        extracted["Enriched"] = True
        self.phones_from_api = self.search(extracted)
        if len(self.phones_from_api) == 0:
            extracted["Enriched"] = False
        # Messy hack to add gsm_arena data
        if "Phone Original" in extracted:
            phone_original = extracted["Phone Original"]
            if phone_original != "" and phone_original in self.gsmarena:
                extracted["Enriched"] = True
                gsmdata = self.gsmarena[phone_original]
                for attr in gsmdata:
                    if extracted[attr] == "":
                        extracted[attr] = gsmdata[attr]

        # elif len(self.phones_from_api) > 1:
        #     # print(json.dumps(self.phones_from_api, indent=4))
        #     pass
        # elif len(self.phones_from_api) == 1:
        #     # print(json.dumps(self.phones_from_api[0], indent=4))
        #     pass
        extracted["Operating System"] = self.get_os()
        extracted["Network Connections"] = self.get_networks()
        capacity = extracted["Storage Capacity"]
        (capacity, memory) = self.get_cap_n_mem(capacity)
        extracted["Storage Capacity"] = capacity
        extracted["Memory RAM"] = memory
        extracted["Phone Screen Size"] = self.get_size()
        features = set(self.get_features())
        for f in extracted["Features"]:
            features.add(f)
        extracted["Features"] = list(features)

        # HACK
        extracted = self.hack(extracted)
        return extracted

    def hack(self, extracted):
        """Tailor results for Shopee..."""
        if "Phone Original" in extracted:
            extracted["Phone Model"] = extracted["Phone Original"]
        else:
            extracted["Phone Model"] = ""
        extracted["Memory RAM"] = ""
        # if extracted["Phone Model"] in self.not_trained_devices:
        #     # print(extracted["Phone Model"])
        #     # print(json.dumps(extracted, indent=4))
        #     return extracted
        extracted["Network Connections"] = ""
        extracted["Phone Screen Size"] = ""
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

    def get_size(self):
        phones = self.phones_from_api
        if len(phones) != 1 or "size" not in phones[0]:
            return ""
        size_str = phones[0]["size"].lower()
        matches = re.match(r"([^\ ]*)\ inches", size_str, re.MULTILINE)
        if matches:
            groups = list(matches.groups())
            if len(groups) == 1:
                size = float(groups[0])
                if (size <= 3.5):
                    return "less than 3.5 inches"
                elif (size >= 3.6 and size <= 4):
                    return "3.6 to 4 inches"
                elif (size >= 4.1 and size <= 4.5):
                    return "4.1 to 4.5 inches"
                elif (size >= 4.6 and size <= 5):
                    return "4.6 to 5 inches"
                elif (size >= 5.1 and size <= 5.5):
                    return "5.1 to 5.5 inches"
                else:
                    return "more than 5.6 inches"
        return ""

    def get_features(self):
        phones = self.phones_from_api
        features = []
        if len(phones) != 1:
            return features
        phone = phones[0]
        if "card_slot" in phone and phone["card_slot"] != "No":
            features.append("expandable memory")
        if "type" in phone and \
                self.string_found(
                        "touchscreen",
                        phone["type"].replace(",", "").lower()):
            features.append("touchscreen")
        if "sensors" in phone and \
                self.string_found(
                        "fingerprint",
                        phone["sensors"].replace(",", "").lower()):
            features.append("fingerprint sensor")
        if "body_c" in phone and \
                self.string_found(
                        "dust",
                        phone["body_c"].replace(",", "").lower()):
            features.append("dustproof")
        if "body_c" in phone and \
                self.string_found(
                        "dustproof",
                        phone["body_c"].replace(",", "").lower()):
            features.append("dustproof")
        if "body_c" in phone and \
                self.string_found(
                        "water/dust",
                        phone["body_c"].replace(",", "").lower()):
            features.append("dustproof")
            features.append("waterproof")
        if "body_c" in phone and \
                self.string_found(
                        "dust/water",
                        phone["body_c"].replace(",", "").lower()):
            features.append("dustproof")
            features.append("waterproof")
        if "body_c" in phone and \
                self.string_found(
                        "water proof",
                        phone["body_c"].replace(",", "").lower()):
            features.append("waterproof")
        if "body_c" in phone and \
                self.string_found(
                        "waterproof",
                        phone["body_c"].replace(",", "").lower()):
            features.append("waterproof")
        if "wlan" in phone and \
                self.string_found(
                        "wi-fi",
                        phone["wlan"].replace(",", "").lower()):
            features.append("wifi")
        if "gps" in phone and \
                self.string_found(
                        "yes",
                        phone["gps"].replace(",", "").lower()):
            features.append("gps")
        return features

    def string_found(self, substr, mainstr):
        substr = " " + substr.strip() + " "
        mainstr = " " + mainstr.strip() + " "
        return substr in mainstr
