# -*- coding: utf-8 -*-
from base64 import encode

import pdfplumber as pdfreader
import json
from local_build.sandbox.ocr.ocr_analysis import image_read_lang


#pdf_file = "..\\res\\class.pdf"
pdf_file = "..\\res\\12578.pdf"


def read_pdf():
    with pdfreader.open(pdf_file) as file:
        pages = file.pages
        print('Pages:\t', pages)
        for page in pages:
            text = page.extract_text()
            print('Extracted Text:\t', text)
        return pages


def output_format(str):
    # For JSON format inptu type
    data = json.dumps(str, indent=4)
    #data = encode(data, 'utf-8')
    #f = open(data.json)
    #data = json.load(f)
#
    print('data:\t', data)
    return data

#read_pdf()
output_format(image_read_lang('C:\\Users\\002CSC744\\Downloads\\pdf_use_case\\photo_6109267155661992742_y.jpg'))