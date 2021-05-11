


###########################################################
import os
import glob
import sys
import numpy as np
import cv2

STUDENT_CODE = '2019193016'
FILE_NAME = 'output.txt'
if not os.path.exists(STUDENT_CODE) :
    os.mkdir(STUDENT_CODE)
f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')