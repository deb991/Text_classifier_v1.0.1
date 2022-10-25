import os
import json

filepath = "C:\\Users\\002CSC744\\Documents\\My_Projects" \
           "\\JText-classifier_main\\config.json"


def read_config():
    file_read = open(filepath)
    data = json.loads(file_read.read())
    # To check if config file readable.

    print("file_data:\t", data["APPLOG"])

    applog = data["APPLOG"]

    return applog


read_config()