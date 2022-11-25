from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import dlib
REAL_SOURCE_DATA = "realimgs"
FAKE_SOURCE_DATA = "fakeimgs"
REAL_OUTPUT_DATA = "real"
FAKE_OUTPUT_DATA = "fake"

def extract(source_data, output_data):
    c = 0

    hog_face_detetor = dlib.get_frontal_face_detector()
    for i in tqdm(os.listdir(source_data)):

        img = cv2.imread(source_data+"/"+i)

        detections = hog_face_detetor(img, 1)

        for face in detections:
            x = face.left()
            y = face.top()
            r = face.right()
            b = face.bottom()

            face_rect = img[np.maximum(y-10, 0):b+10, np.maximum(x-10, 0):r+10]
            if face_rect is not None:
                imgf = cv2.resize(face_rect, (299, 299))
                cv2.imwrite(output_data+"/"+i, imgf)
                c += 1

if __name__ == "__main__":
    extract(REAL_SOURCE_DATA, REAL_OUTPUT_DATA)
    extract(FAKE_SOURCE_DATA, FAKE_OUTPUT_DATA)
