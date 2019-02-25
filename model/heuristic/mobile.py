import utils.envs as env
import json
import collections


class Predictor:
    """Predicts attributes for mobile."""

    def __init__(self):
        self.profiles = self.load_profiles()
        self.bahasa_colors = self.load_bahasa_colors()

    def predict(self, title):
        """Return high-probability attributes as dict.

        All attrs are returned, but may have blank values.
        'Features' is list.
        Other attrs are string, with exception of 'Camera'.
        'Camera' is tentatively list, and this may change.
        """
        attrs = {
                "Features": [],
                "Camera": [],
                "Operating System": "",
                "Network Connections": "",
                "Memory RAM": "",
                "Brand": "",
                "Warranty Period": "",
                "Storage Capacity": "",
                "Color Family": "",
                "Phone Model": "",
                "Phone Screen Size": ""
                }
        remaining = title.lower()
        (remaining, attrs["Color Family"]) = self.extract_color(remaining)
        # print(remaining)
        return attrs

    def extract_color(self, remaining):
        extracted_color = ""
        known_colors = self.profiles["Color Family"]
        for color in known_colors:
            if color in remaining:
                extracted_color = color
                remaining = remaining.replace(color, "")
                return remaining, extracted_color
        for behasa, eng in self.bahasa_colors.items():
            if behasa in remaining:
                extracted_color = eng
                remaining = remaining.replace(behasa, "")
        return remaining, extracted_color

    def load_profiles(self):
        profiles = {}
        given_data = json.loads(
                open(env.mobile_profile_json, "r").read()
                )
        for k, v in given_data.items():
            profiles[k] = list(v.keys())
        # Overwrite colors because of compound names
        profiles["Color Family"] = [
                "navy blue", "light blue", "rose gold",
                "dark grey", "army green", "deep blue",
                "light grey", "deep black", "off white",
                "blue", "gold", "brown", "yellow", "neutral",
                "silver", "pink", "gray", "purple", "rose",
                "multicolor", "black", "apricot", "orange",
                "green", "white", "red"
                ]
        return profiles

    def load_bahasa_colors(self):
        bahasa_colors = collections.OrderedDict()
        bahasa_colors["berwarna mera muda"] = "pink"
        bahasa_colors["biru laut"] = "navy blue"
        bahasa_colors["biru muda"] = "light blue"
        bahasa_colors["mawar emas"] = "rose gold"
        bahasa_colors["abu-abu gelap"] = "dark grey"
        bahasa_colors["hijau tentara"] = "army green"
        bahasa_colors["biru tua"] = "deep blue"
        bahasa_colors["abu-abu terang"] = "light grey"
        bahasa_colors["hitam pekat"] = "deep black"
        bahasa_colors["putih pucat"] = "off white"
        bahasa_colors["beraneka warna"] = "multicolor"
        bahasa_colors["warna jingga"] = "orange"
        bahasa_colors["biru"] = "blue"
        bahasa_colors["emas"] = "gold"
        bahasa_colors["coklat"] = "brown"
        bahasa_colors["kuning"] = "yellow"
        bahasa_colors["netral"] = "neutral"
        bahasa_colors["perak"] = "silver"
        bahasa_colors["abu-abu"] = "gray"
        bahasa_colors["ungu"] = "purple"
        bahasa_colors["mawar"] = "rose"
        bahasa_colors["hitam"] = "black"
        bahasa_colors["aprikot"] = "apricot"
        bahasa_colors["hijau"] = "green"
        bahasa_colors["putih"] = "white"
        bahasa_colors["merah"] = "red"
        return bahasa_colors

    def fake_example(self):
        attrs = {
                "Features": ["fingerprint sensor", "touchscreen"],
                "Camera": ["16mp", "2 mp"],
                "Operating System": "ios",
                "Network Connections": "",
                "Memory RAM": "",
                "Brand": "apple",
                "Warranty Period": "",
                "Storage Capacity": "",
                "Color Family": "",
                "Phone Model": "",
                "Phone Screen Size": ""
                }
        return attrs
