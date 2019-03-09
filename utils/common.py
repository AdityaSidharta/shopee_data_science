import os
import datetime


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_datetime():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_date():
    return datetime.datetime.now().strftime('%Y%m%D')
