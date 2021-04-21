#!/usr/bin/env python3

import os
import csv
import random
import signal
from pathlib import Path

import pandas as pd
import cv2
import tensorflow as tf

from config_nvidia import *
#from utils_end_end import *

def load_lane_infor_data():
    file_name = os.path.join(TRAINING_CSV_FILE_DIR, LANE_DATA_FILE)
    col_list = ["filename", "laneinfor"]
    df = pd.read_csv(file_name, usecols=col_list)
    return df


#def find_file_number(filestr, filename):


def load_training_data(df):
    X = []
    Y = []
    with open(os.path.join(TRAINING_DATA_DIR, TRAINING_DATA_FILE), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for center, left, right, steering, _, _, _ in reader:
            sa = float(steering)
            center = center.replace('hbzhang', 'csu')
            left = left.replace('hbzhang', 'csu')
            right = right.replace('hbzhang', 'csu')

            lane_infor = []
            for index, row in df.iterrows():
                result = row['filename'].find('_')
                if center.find(row['filename'][result:])!= -1:
                    print(row['filename'], row['laneinfor'])
                    lane_infor = row['laneinfor']
                    df.drop(index)
                    break

            X.extend(lane_infor) #[center.strip(), left.strip(), right.strip()])
            Y.extend([sa,
                      sa + ANGLE_DELTA_CORRECTION_LEFT,
                      sa + ANGLE_DELTA_CORRECTION_RIGHT])
    return X, Y


if __name__ == "__main__":
    df = load_lane_infor_data()
    #data = df[0]
    #print(data)

    X, Y = load_training_data(df)






