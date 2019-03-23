import json
import re
regex = re.compile('[^a-zA-Z ]')


def load_library(filename):
    with open(filename, 'r+') as file:
        feature_profile = json.load(file)
    return feature_profile


def get_first(mylist):
    return mylist[0]


def get_second(mylist):
    return mylist[1]


def _word_compare(texts_lib, library):
    for text in texts_lib:
        value = library.get(text, None)
        if not value is None:
            return value


def _word_compare_ans(texts_lib, library):
    for text in texts_lib:
        value = library.get(text, None)
        if not value is None:
            return value, text
    return None, None

        
def _split_texts(title, level):
    texts = title.split()
    texts_lib = texts
    texts_temp = texts
    for i in range(1, level):
        texts_temp = list(zip(texts_temp, texts[i:]))
        texts_temp = [' '.join(text) for text in texts_temp]
        texts_lib += texts_temp
    return texts_lib


def word_compare(title, libraries, level=4):
    texts_lib = _split_texts(title, level)

    if isinstance(libraries, list):
        for lib in libraries:
            value = _word_compare(texts_lib, lib)
            if not value is None:
                return value
        return None
    else:
        return _word_compare(texts_lib, libraries)


def word_compare_ans(title, libraries, level=4):
    texts_lib = _split_texts(title, level)

    if isinstance(libraries, list):
        for lib in libraries:
            value, text = _word_compare_ans(texts_lib, lib)
            if not value is None:
                return value, text
        return None, None
    else:
        return _word_compare_ans(texts_lib, libraries)


def get_feature(title, library, level=4):
    res = word_compare(
        title,
        libraries=library,
        level=level
    )
    return res


def get_feature_ans(title, library, level=4):
    res, keyword = word_compare_ans(
        title,
        libraries=library,
        level=level
    )
    return res, keyword


def get_feature_strip(title, library, level=4):
    res, key = get_feature_ans(title, library, level)
    if res is not None:
        title = title.replace(key, "-")
        return [title, res]
    return [title, None]


def get_feature_wo_num(title, library, level=4):
    title = regex.sub('', title)
    res, keyword = word_compare_ans(
        title,
        libraries=library,
        level=level
    )
    return [title, res]
