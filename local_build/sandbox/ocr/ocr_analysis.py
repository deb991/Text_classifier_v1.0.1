import cv2
import os, argparse
import pytesseract as ocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from time import sleep
import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

custom_config =  r'--oem 3 --psm 6'
logger.info('Custom Config set to: --oem 3 --psm 6')
ocr.pytesseract.tesseract_cmd = \
    'C:\\Tesseract-OCR\\tesseract.exe'
logger.info('Py Tessaract executable path set. ')


def detect_image_lang(image):
    logger.info(' Image language recognization initiated. ')
    try:
        osd = ocr.image_to_osd(image)
        script = re.search("Script: ([a-zA-Z]+)\n", osd)[1]
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd)[1]
        #print('Script:\t', script)
        return script, float(conf)
    except Exception:
        logger.error(' Exception cause ended language detection. ')
        return None, 0.0


def image_read(image):
    logger.info('Test image read initiated. ')
    img = cv2.imread(image)
    raw_data = ocr.image_to_string(img, config=custom_config)

    return raw_data


def lang_classifier(arg1):
    logger.info('Language Classifier from image text initiated. ')
    try:
        if arg1 == 'english':
            return 'eng'
        elif arg1 == 'Bengali':
            return 'ben'
    except:
        logger.exception('Language classification exception!! ')
        print('Language detectection not happen!!')


def image_read_lang(image):
    logger.info('Lang classification basis read data initiated. ')
    script_name, confidence = detect_image_lang(image)
    get_lang = lang_classifier(script_name)
    im = Image.open(image)
    im = im.filter(ImageFilter.MedianFilter)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    text = ocr.image_to_string(Image.open(image), lang=get_lang)
    #print('Text_check:\t', text)
    logger.info('Return language based data')
    return text


def np_array_img(image):
    logger.info('Numpy array conversion & return. ')
    np_image = np.array(Image.open(image))
    return np_image


def get_greyscale(image):
    logger.info(" 'greyscale' data of extracted image text. ")
    grey_img = cv2.cvtColor(np_array_img(image), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Geay_Scale_window", grey_img)
    cv2.waitKey(30)
    #sleep(5)
    #return cv2.cvtColor(np_array_img(image), cv2.COLOR_BGR2GRAY)


def get_noise_removal(image):
    logger.info(' Noise removal initiated. ')
    noise_remove_img = cv2.medianBlur(np_array_img(image), 5)
    cv2.imshow("Noise_Removed_image", noise_remove_img)
    cv2.waitKey(30)
    #sleep(5)
    #return noise_remove_img


def get_thresholding(image):
    logger.info(' get threshold. ')
    float_img = np.random.random((4, 4))
    im = np.array(float_img * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 3, 0)
    #img_threshold = cv2.threshold(np_array_img(image), 0, 255,
    #                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresold_Image", threshed)
    cv2.waitKey(30)
    #sleep(5)
    #return threshed


def get_dialation_img(image):
    logger.init(" get dialation of the inout image. ")
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(np_array_img(image), kernel, iterations=1)


def get_erosion(image):
    logger.info(' get Erossion of the inout image. ')
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(np_array_img(image), kernel, iterations=1)


def opening_image(image):
    logger.info("Image open process initiated. ")
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(np_array_img(image), cv2.MORPH_OPEN, kernel)


def get_canny_edge(image):
    logger.info('Canny Image initated. ')
    return cv2.Canny(np_array_img(image), 100, 200)


def get_skew_correction(image):
    logger.info(" Skew correction process initiated. ")
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        cm = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, cm, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        cv2.imshow('Rotated_Image', rotated)
        cv2.waitKey(30)
        #sleep(5)
        #return rotated


def get_match_template(image, template):
    logger.info("Match template initiated. ")
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def ocr_test(image):
    logger.info(" OCR test initiated with inout function. ")
    script_name, confidence = detect_image_lang(image)
    #print(f'Script_Name: \t{script_name}, \nConfindence:\t{confidence}')
    image_read_lang(image)
    image_read(image)
    get_greyscale(image)
    get_noise_removal(image)
    get_thresholding(image)
    sleep(60)
    #get_dialation(image)
    get_canny_edge(image)

    #return image_data, noise_removed, thresholding, dialation, edge_detectin


if __name__ == '__main__':
    #ocr_test('C:\\Users\\002CSC744\\Documents'
    #         '\\My_Projects\\JText-classifier_main\\image\\12578\\Page_12.jpg')
    ocr_test("C:\\Users\\002CSC744\\Downloads\\pdf_use_case\\photo_6109267155661992742_y.jpg")