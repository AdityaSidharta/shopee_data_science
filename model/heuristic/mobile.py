import utils.envs as env

def predict(title):
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
    return fake_example()

def fake_example():
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
