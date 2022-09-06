import os
import fitz
import pytesseract
import cv2
import io
from PIL import Image, ImageFile, ImageTk  # from Pillow
import pdfplumber as native_pdf_reader
from dagster import op, job
import enum
import wx


custom_config =  r'--oem 3 --psm 6'

file1 = 'C:\\Users\\002CSC744\\Documents\\My_Projects' \
            '\\JText-classifier_main\\res\\NatGeo_1.pdf'
file2 = "C:\\Users\\002CSC744\\Documents\\My_Projects" \
            "\\JText-classifier_main\\res\\class.pdf"
file3 = "C:\\Users\\002CSC744\\Documents\\My_Projects" \
            "\\JText-classifier_main\\res\\12578.pdf"


def scanned_text_extract(page_list: list):
    text_list = []
    for i in page_list:
        page = i
        text_ext = page.get_text("text")
        #print('Checking tesxt:\t', text_ext)
        if len(text_ext) == 0:
            print('\t\t\t\tNeed to initiate OCR to extract Data')
        else:
            text_list.append(text_ext)
    return text_list


def scanned_img_extract(page_list: list):
    images = []
    for i in page_list:
        page = i
        img_ext = page.get_pixmap()
        images.append(img_ext)
    print('Image List:\t', images)
    return images


def get_scanned_pdf(args):
    pages = []
    image_list = []
    text_block = []
    pdf_file = fitz.open(args)
    #print('PDF_FILE:\t', pdf_file)
    #page = pdf_file[8]
    #print('\nPage:\t', page)
    for page in pdf_file:
        pages.append(page)
    #print('Pages:\t', pages)

    text_block.append(scanned_text_extract(pages))
    #print("Text paragraph:\t", text_block)

    image_list.append(scanned_img_extract(pages))
    #print('\nImage_List:\t', image_list)


def get_native_pdf_data(args):
    with native_pdf_reader.open(args) as file:
        pages = file.pages
        print('Pages:\t', pages)
        for page in pages:
            text = page.extract_text()
            print('Extracted Text:\t', text)
        return pages


if __name__ == '__main__':
    get_scanned_pdf(file1)
