import cv2
import numpy as np
import os

READ_IMAGES_PATH = "/Users/filipgulan/Diplomski/Seminar/Slova/"
WRITE_IMAGES_PATH = "/Users/filipgulan/Diplomski/Seminar/Obrada/"

for root, dirs, files in os.walk(READ_IMAGES_PATH):
    for file in files:
        if file.endswith('.png'):
            image_path = os.path.join(root, file)
            folder_name = root.replace(READ_IMAGES_PATH, '')

            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_image_path = WRITE_IMAGES_PATH + folder_name
            if not os.path.exists(new_image_path):
                os.makedirs(new_image_path)
            print(os.path.join(new_image_path, file))

            (thresh, bw_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bw_image = cv2.bitwise_not(bw_image)
            contours = cv2.findContours(bw_image ,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imwrite(os.path.join(new_image_path, file), bw_image)
