import utils.envs as env
import json
import collections


class Extractor:
    """Extracts attributes for mobile from title."""

    def __init__(self):
        self.profiles = self.load_profiles()
        self.bahasa_colors = self.load_bahasa_colors()

    def extract_from_title(self, title):
        """Return high-probability attributes as dict.

        All attrs are returned, but may have blank values.
        'Features' is list.
        Other attrs are string, with exception of 'Camera'.
        'Camera' is tentatively list, and this may change.
        """
        attrs = {
                "Features": [],
                "Camera": [],
                "Color Family": "",
                "Phone Model": "",
                "Brand": "",
                "Operating System": "",
                "Network Connections": "",
                "Memory RAM": "",
                "Warranty Period": "",
                "Storage Capacity": "",
                "Phone Screen Size": "",
                "Impossible": False,
                "All Text Extracted": False
                }
        remaining = title.lower()
        remaining = self.rm_fillers(remaining)

        # First, is it even a phone?
        if self.should_skip(title):
            attrs["Impossible"] = True
            return attrs

        # Then color
        (remaining, attrs["Color Family"]) = self.extract_color(remaining)

        # Then model + brand
        extracted_phone = self.extract_phone(remaining)
        remaining = extracted_phone["remaining"]
        attrs["Brand"] = extracted_phone["Brand"]
        attrs["Phone Model"] = extracted_phone["Phone Model"]

        # Then brand
        if attrs["Brand"] == "":
            (remaining, attrs["Brand"]) = self.extract_brand(remaining)

        # Then capacity
        storage = "Storage Capacity" # just to shorten the line
        (remaining, attrs[storage]) = self.extract_storage(remaining)
        print(remaining)

        # Skip RAM. If capacity + device available, that's more useful

        # Then known features
        (remaining, attrs["Features"]) = self.extract_features(remaining)

        if remaining.strip() == "":
            attrs["All Text Extracted"] = True

        return attrs

    def rm_fillers(self, remaining):
        fillers = [
                "and", "anti", "new", "brand", "charger",
                "original", "fast", "promo", "wa", "ini",
                "hari", "discount", "stock", "chat",
                "handphone", "deal", "hot"
                ]
        for filler in fillers:
            if self.string_found(filler, remaining):
                remaining = remaining.replace(filler, "")
        return remaining

    def should_skip(self, remaining):
        skip_names = [
                "drypers",
                "mamypoko",
                "huggies",
                "pampers",
                "macbook pro"
                ]
        for name in skip_names:
            if self.string_found(name, remaining):
                return True
        return False

    def extract_color(self, remaining):
        extracted_color = ""
        for color in self.profiles["Color Family"]:
            if self.string_found(color, remaining):
                extracted_color = color
                remaining = remaining.replace(color, "")
                return remaining, extracted_color
        for behasa, eng in self.bahasa_colors.items():
            if self.string_found(behasa, remaining):
                extracted_color = eng
                remaining = remaining.replace(behasa, "")
                break
        return remaining, extracted_color

    def extract_phone(self, remaining):
        extracted = {
                "remaining": remaining,
                "Brand": "",
                "Phone Model": ""
                }
        for brand, phone in self.profiles["Phone Models - Edited"]:
            phone_str = " ".join([brand, phone])
            if self.string_found(phone_str, remaining):
                extracted["Brand"] = brand
                extracted["Phone Model"] = phone
                extracted["remaining"] = remaining.replace(
                        phone_str,
                        ""
                        )
                break
            elif len(phone) > 3 and self.string_found(phone, remaining):
                extracted["Brand"] = brand
                extracted["Phone Model"] = phone
                extracted["remaining"] = remaining.replace(
                        phone,
                        ""
                        )
                break
        return extracted

    def extract_brand(self, remaining):
        extracted_brand = ""
        for brand in self.profiles["Brand"]:
            if self.string_found(brand, remaining):
                extracted_brand = brand
                remaining = remaining.replace(brand, "")
                break
        return remaining, extracted_brand

    def extract_storage(self, remaining):
        extracted_storage = ""
        for capacity in self.profiles["Capacities - Edited"]:
            if self.string_found(capacity, remaining):
                remaining = remaining.replace(capacity, "")
                extracted_storage = capacity
                extracted_storage = extracted_storage.replace(" ", "")
                break
        return remaining, extracted_storage

    def extract_features(self, remaining):
        extracted = []
        for feature in self.profiles["Features"]:
            if self.string_found(feature, remaining):
                extracted.append(feature)
                remaining = remaining.replace(feature, "")
        return remaining, extracted

    def load_profiles(self):
        profiles = {}
        given_data = json.loads(
                open(env.mobile_profile_json, "r").read()
                )
        for k, v in given_data.items():
            profiles[k] = list(v.keys())

        # Fix phone models to separate their brand names
        replace_models = []
        skip_devices = [
                "a6000",
                "6",
                "a39",
                "z2",
                "a5000",
                "105"
                ]
        for phone_model in profiles["Phone Model"]:
            brand = phone_model.split()[0]
            device = " ".join(phone_model.split()[1:])
            if device in skip_devices:
                continue
            if brand in profiles["Brand"]:
                for device_perm in self.str_permutations(device):
                    replace_models.append((brand, device_perm))
            else:
                for device_perm in self.str_permutations(phone_model):
                    replace_models.append(("", device_perm))
        replace_models.sort(key=lambda x: len(x[1]), reverse=True)
        profiles["Phone Models - Edited"] = replace_models

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

        # These brands are not in the given profiles
        profiles["Known Missing Brands"] = [
                "mobiistar", "toshiba", "canon", "genki",
                "niko", "motorola", "panasonic", "meizu",
                "dell", "nikon", "msi", "olympus", "sensi",
                "ichiko", "changhong", "fujifilm", "leica",
                "sanken"
                ]

        # These are all the possible capacities:
        profiles["Capacities - Edited"] = [
                "512gb", "512 gb", "256gb", "256 gb", "128gb",
                "128 gb", "64gb", "64 gb", "32gb", "32 gb",
                "16gb", "16 gb", "10gb", "10 gb", "8gb", "8 gb",
                "6gb", "6 gb", "4gb", "4 gb", "3gb", "3 gb",
                "2gb", "2 gb", "1.5gb", "1.5 gb", "1gb", "1 gb",
                "512mb", "512 mb", "256mb", "256 mb", "128mb",
                "128 mb", "4mb", "4 mb"
                ]

        return profiles

    def str_permutations(self, phone):
        """For example, note4 == note 4."""
        perms = [phone]
        phone = " " + phone + " "
        for i in range(1, 10):
            i_str = " " + str(i) + " "
            if i_str in phone:
                perm_str = phone.replace(
                        i_str,
                        str(i) + " "
                        )
                perms.append(perm_str.strip())
        return perms

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
        # Hack...
        bahasa_colors["navyblue"] = "navy blue"
        bahasa_colors["lightblue"] = "light blue"
        bahasa_colors["rosegold"] = "rose gold"
        bahasa_colors["darkgrey"] = "dark grey"
        bahasa_colors["armygreen"] = "army green"
        bahasa_colors["deepblue"] = "deep blue"
        bahasa_colors["lightgrey"] = "light grey"
        bahasa_colors["deepblack"] = "deep black"
        bahasa_colors["offwhite"] = "off white"
        bahasa_colors["dark gray"] = "dark grey"
        bahasa_colors["darkgray"] = "dark grey"
        bahasa_colors["grey"] = "gray"
        return bahasa_colors

    def string_found(self, substr, mainstr):
        substr = " " + substr.strip() + " "
        mainstr = " " + mainstr.strip() + " "
        return substr in mainstr
