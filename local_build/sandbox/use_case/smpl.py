import os
from local_build.sandbox.ocr.ocr_analysis import image_read_lang


file = "C:\\Users\\002CSC744\\Downloads" \
       "\\pdf_use_case\\photo_6109267155661992740_y.jpg"

data = image_read_lang(file)
print('Scanned Data:\t', data)

