import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config_lstr import system_configs

from utils import crop_image, normalize_

from sample.vis import *

from PIL import Image

import pandas as pd

import csv

class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)
        return results

def kp_detection(db, nnet, image_root, debug=False, evaluator=None):
    input_size  = db.configs["input_size"]  # [h w]
    image_dir = os.path.join(image_root, "images")
    result_dir = os.path.join(image_root, "detections")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_names = os.listdir(image_dir)
    num_images = len(image_names)

    #num_images = 1

    postprocessors = {'bbox': PostProcess()}

    length = range(0, num_images)

    lane_infor_file = "lane_infor_file.csv"
    lane_data = []

    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        image_file    = os.path.join(image_dir, image_names[ind])
        image         = cv2.imread(image_file)
        height, width = image.shape[0:2]

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        #prefilled_images = np.zeros((1, 3, height, width), dtype=np.float32)
        prefilled_images = np.zeros((height, width, 3), np.uint8)
        #prefilled_images = np.zeros([height, width], np.uint8)

        pad_image     = image.copy()
        #pad_image     = prefilled_images

        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255.
        normalize_(resized_image, db.mean, db.std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t0            = time.time()
        outputs, weights      = nnet.test([images, masks])
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t             = time.time() - t0
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if evaluator is not None:
            evaluator.add_prediction(ind, results.cpu().numpy(), t)

        if debug:
            pred = results[0].cpu().numpy()
            #img  = prefilled_images #pad_image
            img =  pad_image
            img_h, img_w, _ = img.shape
            #img_h, img_w = img.shape
            pred = pred[pred[:, 0].astype(int) == 1]
            overlay = img.copy()
            color = (255, 0, 0)

            fileName = os.path.join(result_dir, image_names[ind][:-4] + '.jpg')

            cv2.imwrite(fileName, img)

            drawPloy(img, pred, img_h, img_w,fileName, lane_data)

            for i, lane in enumerate(pred):
                # draw lane ID
                #if len(points) > 0:
                #    cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #               color=color,
                #                thickness=1)
                drawLine(overlay, i, pred, lane, img_h, img_w, color)

            # Add lanes overlay
            w = 0.6
            #img = ((1. - w) * img + w * overlay).astype(np.uint8)

            fileName_overlay = os.path.join(result_dir, image_names[ind][:-4] + '.jpg')

            #print("Image file saved {}".format(fileName))

            #cv2.imwrite(fileName_overlay, img)

    csv_columns = ['filename', 'laneinfor']
    try:
        with open(lane_infor_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in lane_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    #pd.DataFrame(lane_data).to_csv(lane_infor_file)

    return 0


def drawPloy(img, pred, img_h, img_w,fileName, lane_data):

    #print('image shape is %s' % (str(img.shape)))
    #print("pred size {}".format(pred.shape))
    points = np.zeros([4,1])
    masked_image = img #np.zeros((img_h, img_w, 3), np.uint8)
    for i, lane in enumerate(pred):#pred[:-1]):
        lane = pred[i]
        #lane1 = pred[i+1]

        pt1, pt2 = identifyPoints(i, lane, img_h, img_w)

        #pt3, pt4 = identifyPoints(i, lane1, img_h, img_w)

       # print(" iteration {} pt1 {}, pt2 {}".format(i, pt1,pt2))

        pt1_row = np.row_stack([pt1.reshape([2,1]), pt2.reshape([2,1])])

        #pt2_row = np.row_stack([pt3.reshape([2,1]), pt4.reshape([2,1])])

        #point = np.column_stack((pt1_row, pt2_row))

        points = np.append(points, pt1_row,axis=1)

    if pred is not None and len(pred) > 1:
        lane_current = pred[0]
    else:
        lane_current = np.zeros([9])



    extract_middle_lane(points, lane_current, fileName, lane_data)

    #print("Points shape {} ".format(points.shape))
    masked_image = identifyNeighbour_fill_solid(img,  img_h, img_w, points)

    # print("Points shape {} ".format(points.shape))
    #masked_image = identifyNeighbour(img, img_h, img_w, points)


    # The resultant image
    cv2.imwrite(fileName, masked_image)

def extract_middle_lane(pts, lane, fileName, lane_data):
    row = {}
    row['filename'] = fileName
    row['laneinfor'] = lane
    lane_data.append(row)


def identifyNeighbour(img,  img_h, img_w, pts):
    #print("before sort pts array is {}".format(pts))

    pts = selectionSort(pts)

    mask = np.ones(img.shape, dtype=np.uint8)
    mask.fill(255)

    #print("after sort pts array is {}".format(pts))

    masked_image = img #np.zeros((img_h, img_w, 3), np.uint8)
    for i, pt in enumerate(pts[:,1:pts.shape[1]-1].T):

        '''
          print("index i is {} pts shape".format(i, pts.shape[1]))
        print("pt1 {}, pt2 {}, pt3 {}, pt4 {}".format(

            [pts[0, i+1], pts[1, i+1]], [pts[0, i + 2], pts[1, i + 2]],
            [pts[2, i + 2], pts[3, i + 2]], [pts[2, i+1], pts[3, i+1]]

        ))
        
        '''


        a = np.array([[ [pts[0,i+1], pts[1,i+1]],[pts[0,i+2], pts[1,i+2]],
                        [pts[2, i+2], pts[3, i+2]], [pts[2, i+1], pts[3, i+1]]
                        ]], dtype=np.int32)



        # print("lane {} pojnts {} ".format(i, a))

        # a = np.array([[[10, 10], [100, 10], [10, 100], [100, 100]]], dtype=np.int32)

        #cv2.fillPoly(img, a, 255 - (i+1) * 100)

        #color = [60, 0, 255]

        color = [0, 0, 0]

        cv2.fillPoly(mask, a, color)

        masked_image = cv2.bitwise_or(img, mask)

    return masked_image
        # print(str(pred[i]))


def identifyNeighbour_fill_solid(img, img_h, img_w, pts):
    # print("before sort pts array is {}".format(pts))

    pts = selectionSort(pts)
    # print("after sort pts array is {}".format(pts))

    for i, pt in enumerate(pts[:, 1:pts.shape[1] - 1].T):
        '''
          print("index i is {} pts shape".format(i, pts.shape[1]))
        print("pt1 {}, pt2 {}, pt3 {}, pt4 {}".format(

            [pts[0, i+1], pts[1, i+1]], [pts[0, i + 2], pts[1, i + 2]],
            [pts[2, i + 2], pts[3, i + 2]], [pts[2, i+1], pts[3, i+1]]

        ))

        '''

        a = np.array([[[pts[0, i + 1], pts[1, i + 1]], [pts[0, i + 2], pts[1, i + 2]],
                       [pts[2, i + 2], pts[3, i + 2]], [pts[2, i + 1], pts[3, i + 1]]
                       ]], dtype=np.int32)

        # print("lane {} pojnts {} ".format(i, a))

        # a = np.array([[[10, 10], [100, 10], [10, 100], [100, 100]]], dtype=np.int32)

        cv2.fillPoly(img, a, 255 - (i+1) * 100)

        #color = [60, 0, 255]

        #color = [0, 0, 0]

        #cv2.fillPoly(mask, a, color)

        #masked_image = cv2.bitwise_or(img, mask)

    return img

def selectionSort(pts):

    sorted_pts = []
    sorted_pts = pts[:,pts[0,:].argsort()]

    return sorted_pts

def selectionSortFromScratch(pts):

    xCoordinate = np.zeros([pts.shape[1]])

    for i, value in enumerate(pts[0,:]):
        xCoordinate[i] = pts[0, i]
    #ycoordinate = pts[1, :]

    print(" before sort xCoordinate {}".format(xCoordinate))

    for i, y in enumerate(xCoordinate):
        value = xCoordinate[i]
        index = i
        entireValue = pts[:,index]

        for j, z in enumerate(xCoordinate[index+1:]):
            j1 = i + j + 1
            if xCoordinate[j1] < value:
                value = xCoordinate[j1]
                entireValue = pts[:,j1]
                index = j1


        if i != index:
            print("now switching i {} and index {}".format(i,index))
            xCoordinate[index] = xCoordinate[i]
            xCoordinate[i] = value
            pts[:,index] = pts[:,i]
            pts[:,i] = entireValue

    print(" after sort xCoordinate {}".format(xCoordinate))
    print("after sort pts array is {}".format(pts))

def identifyPoints(i, lane, img_h, img_w):
    lane = lane[1:]  # remove conf
    lower, upper = lane[0], lane[1]
    lane = lane[2:]  # remove upper, lower positions

    # generate points from the polynomial
    ys = np.linspace(lower, upper, num=100)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(int)
    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                     lane[5]) * img_w).astype(int)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

    #print("popints shape {}".format(points.shape))
    return points[1,:], points[-2,:]



def drawLine(overlay, i , pred,lane, img_h, img_w,color):
    #print(str(i) + " lane")
    #print(" lane information " + str(lane[0]) + " " + str(lane[1]))
    #print(type(pred))
    lane = lane[1:]  # remove conf
    lower, upper = lane[0], lane[1]
    lane = lane[2:]  # remove upper, lower positions

    # generate points from the polynomial
    ys = np.linspace(lower, upper, num=100)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(int)
    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                     lane[5]) * img_w).astype(int)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

    # print(" points " + str(points[:-1]))

    # draw lane with a polyline on the overlay
    for current_point, next_point in zip(points[:-1], points[1:]):
        #print("drawing line now {} {}".format(current_point, next_point))
        overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=5)

    #return overlay

def testing(db, nnet, image_root, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, image_root, debug=debug, evaluator=evaluator)