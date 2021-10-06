import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False})
import cv2
from numba import jit

def get_bbox_area(col):
  return ((col["x_max"] - col["x_min"]) * (col["y_max"] - col["y_min"]))

def get_bbox_area_normalize(col):
  return ((col["xmax_norm"] - col["xmin_norm"]) * (col["ymax_norm"] - col["ymin_norm"]))


def get_bbox_width(col):
    return ((col["x_max"] - col["x_min"]))


def get_bbox_height(col):
    return ((col["y_max"] - col["y_min"]))

def get_bbox_aspect_ratio(col):
    ratio = round(get_bbox_width(col)/ get_bbox_height(col), 3)
    return ratio


def plot_single_img(img, size = (18,18),is_rgb = True, title = None, cmap = "gray"):
    plt.figure(figsize = size)
    plt.imshow(img, cmap = cmap)
    plt.suptitle(title)
    plt.show()



def plot_multi_imgs(imgs, # 1 batchs contain multiple images
                    cols = 2, size = 10, # size of figure
                    is_rgb = True, title = None, cmap = "gray",
                    img_size = None, name_fig = None, close = False): # set img_size if you want (width, height)
    rows = (len(imgs) // cols) + 1
    fig = plt.figure(figsize = (size *  cols, size * rows))
    for i , img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1) # add subplot int the the figure
        plt.imshow(img, cmap = cmap) # plot individual image
    
    plt.suptitle(title)
    if name_fig != None:
        plt.savefig(name_fig)
    if close:
        plt.close


def draw_bbox(img, # single image
              box, # list 1-D with [x_min, y_min, x_max, y_max] of this image
              label, # string --> name of this bbox
              color, # color of this box
              thickness,
              alpha_inside_bbox = .1,
              alpha_inside_label = .4):
    overlay_bbox = img.copy()  #create a copy image to draw the overlay bbox
    overlay_label = img.copy() #create a copy image to draw the overlay label

    final_image = img.copy() # the function return this image

    text_width, text_height = cv2.getTextSize(label.upper(), fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale= 0.6, thickness= 1)[0]
    # this function will return the width and height of text


    '''Draw the overlay of bbox'''
    # drawn on overlay_bbox
    cv2.rectangle(overlay_bbox,
                 pt1 = (box[0], box[1]),
                 pt2 = (box[2], box[3]),
                 color =  color,
                 thickness= - 1) # -1 --> mean this will overlay the rectangle


    '''Creat the alpha of the overlay rectangle''' #--> using cv2.addWeight()
    cv2.addWeighted(src1 = overlay_bbox,
                   alpha = alpha_inside_bbox,
                   src2 = final_image,
                   beta= 1- alpha_inside_bbox,
                   gamma= 0,
                   dst = final_image)

    '''Draw the overlay of label'''
    cv2.rectangle(overlay_label,
                 pt1 = (box[0], box[1] - 7 - text_height),
                 pt2 = (box[0] + text_width +  2, box[1]),
                 color= (0,0,0), # white text color
                 thickness= - 1) # overlay the rectangle

    '''Creat the alpha of the overlay rectangle'''
    cv2.addWeighted(src1= overlay_label,
                   alpha = alpha_inside_label,
                   src2= final_image,
                   beta= 1 - alpha_inside_label,
                   gamma= 0,
                   dst= final_image)

    '''Draw the thinckess of bbox'''
    cv2.rectangle(final_image,
                 pt1= (box[0], box[1]),
                 pt2 = (box[2], box[3]),
                 color = color,
                 thickness= thickness)

    cv2.putText(final_image,
               text = label.upper(),
               org = (box[0], box[1] - 5),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
               fontScale= 0.6,
               color= (255,255,255),
               thickness= 1, lineType= cv2.LINE_AA)

    return final_image

def draw_bbox_heatmap(df, class_id, heatmap_size, index_ymin, index_ymax, index_xmin, index_xmax):

    # first --> initialize the black background
    heatmap = np.zeros((heatmap_size))
    for _, series in df[df["class_id"] == class_id].iterrows():
        # we add 1 value inside the bboxes
        heatmap[series[index_ymin]: series[index_ymax], series[index_xmin] : series[index_xmax]] +=1

    return heatmap


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # box = [x_min, y_min, x_max, y_max]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def draw_bbox_on_heatmap(bboxes, class_id, heatmap_size):
    # initialize empty (full black) heatmap
    heatmap = np.zeros((heatmap_size))
    for _, row in bboxes[bboxes["class_id"] == class_id].iterrows():
        # draw white bboxes on black heatmap based on bboxes coordinate
        # that mean multiple bboxes which same class_id will be drawn on same heatmap
        heatmap[row[2]:row[4], row[1]:row[3]] += 1
    return heatmap

