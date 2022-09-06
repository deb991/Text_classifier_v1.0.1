import cv2
import os, argparse
import pytesseract as ocr
import numpy as np

custom_config =  r'--oem 3 --psm 6'
ocr.pytesseract.tesseract_cmd = \
    'C:\\Tesseract-OCR\\tesseract.exe'

def image_data(image):
    print('Data:\t', cv2.imread(image))
    return cv2.imread(image)


def image_read(image):
    img = cv2.imread(image)
    raw_data = ocr.image_to_string(img, config=custom_config)
    print('Raw Data:\t', raw_data)
    return raw_data


def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_noise_removal(image):
    return cv2.medianBlur(image, 5)


def get_thresholding(image):
    return cv2.threshold(image, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_dialation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def get_erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening_image(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def get_canny_edge(image):
    return cv2.Canny(image, 100, 200)


def get_skew_correction(image):
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
        return rotated


def get_match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def ocr_test(image):
    data = image_read(image)
    #grayScaling = get_greyscale(image)
    noise_removed = get_noise_removal(image)
    thresholding = get_thresholding(image)
    dialation = get_dialation(image)
    edge_detectin = get_canny_edge(image)

    return image_data, noise_removed, thresholding, dialation, edge_detectin


if __name__ == '__main__':
    ocr_test('C:\\Users\\002CSC744\\Documents'
             '\\My_Projects\\JText-classifier_main\\image\\12578\\Page_12.jpg')
