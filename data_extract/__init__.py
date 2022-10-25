# -*- coding: utf-8 -*-
from base64 import encode
import os
import pdfplumber as pdfreader
import json
from local_build.sandbox.ocr.ocr_analysis import image_read_lang
import logging

#log_file = os.path.basename(read_config())
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

#pdf_file = "..\\res\\class.pdf"
pdf_file = "..\\res\\12578.pdf"

get_str = image_read_lang('C:\\Users\\002CSC744\\Downloads\\pdf_use_case'
                          '\\photo_6109267155661992742_y.jpg')
logger.info(" 'get_string' function extract text from image.")

def read_pdf():
    logger.info(" 'read_pdf' initiated to read pdf file for Data Extraction.")
    logger.info('Read PDF file init.')
    with pdfreader.open(pdf_file) as file:
        pages = file.pages
        print('Pages:\t', pages)
        for page in pages:
            text = page.extract_text()
            print('Extracted Text:\t', text)
        logger.info('Returning Pages')
        return pages


"""Below function is to define whether the string is as per any specific 
file based or matched with any type of file format. This return statement 
is to redirected to LDA model & construct Data Frame to train model. """


def json_convert(img_str: str):
    logger.info(" 'json_convert' initiated to convert input string into "
                 "json format")
    if json.loads(img_str) == True:
        json_data = json.loads(img_str)
    else:
        print('Not a valid Json construct string!!')


def output_format(str):
    data = json.dumps(str, ensure_ascii=False).encode('utf8')
    logger.info('Decoder initiated for str> Json file sttructure. ')
    data = data.decode()
    json_strct = json.loads(data)
    data = json_strct
    logger.info('Json Data recognized. ')
    #print('Json Structure validation:\t', data)

    logger.info("Return Json data for data frame. ")
    return data

#read_pdf()
output_format(get_str)