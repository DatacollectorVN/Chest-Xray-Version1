import seaborn as sns
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False})
import numpy as np
import pandas as pd
import os
import cv2
from tqdm.auto import tqdm 
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import argparse
from utils_DP import *
from ensembles_boxes_custom import ensemble_boxes_NMS

base_dir = os.path.join('..', '..')
dataset = 'Dataset'
dataset_dir = os.path.join(base_dir, dataset)
Intermediates_dir = 'Intermediates'
Charts_Images_dir = os.path.join(Intermediates_dir, 'Charts_Images')
csv_txt_json_data = os.path.join(Intermediates_dir, 'csv_txt_json_data')

IOU_THR = 0.3
SKIP_BOX_THR = 0.001
THICKNESS = 3
SIGMA = 0.1

label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100], [0,0,0]]

def load_data(path_csv):
    '''
    :param path_csv: file_csv inside csv_txt_json_data directory
    :return: dataframe  
    '''
    df =  pd.read_csv(os.path.join(csv_txt_json_data, path_csv), index_col= 0)
    return df

def GetKey(val, dct_class_ids):
   for key, value in dct_class_ids.items():
      if val == value:
        
        return key
      continue

def main(path_csv= "dataset_after_processing_fe.csv", mode = 1, start = 0, stop= 5,
         method = 3, file_save = 'after_remove.csv', fig_save = 'result_NMS.png',
         iou_thr = IOU_THR, skip_box_thr = SKIP_BOX_THR, thickness = THICKNESS, sigma = SIGMA):
    '''
    :param: mode: 
        + mode = 1 (Remove and display image) 
        --> NMS or soft-NMS several images and save the results 
        to Intermediates_dir/Charts_Images_dir folder
        + mode = 2 (Remove and save)
        --> NMS or soft-NMS all images in Datasets folder and save all information into 
        Intermediates_dir/csv_txt_json_data folder
    :param: method: (defaul = 3)
        method = 1 --> soft-NMS with linear penalty
        method = 2 --> soft-NMS with Gaussian penalty
        method = 3 --> NMS
    '''
    df_train = load_data(path_csv= path_csv)
    df_finding_train = df_train[df_train["class_id"] !=14]
    
    x = df_finding_train[["class_name", "class_id"]].groupby(by = ["class_name", "class_id"]).count().reset_index()
    dct_class_ids = {}
    for i in x.to_numpy():
        dct_class_ids[i[0]] = int(i[1])
        dct_class_ids["No finding"] = 14

    imgs_path = df_finding_train.image_path.unique()
    viz_imgs = list() # the list to contain all images that will visualize
    
    # MODE = 1 REMOVE AND DISPLAY SEVERAL IMAGES
    # default start_mode_1 = 0 and end_mode_1 = 5 --> manipulate with first 5 images
    if mode == 1:
        for i, img_path in tqdm(enumerate(imgs_path[start:stop]), total = len(imgs_path[start:stop])):
            img_array = cv2.imread(img_path) # read the image (without bboxes)
            img_id = Path(img_path).stem
            print(f"img_id = {img_id}")
            img_annotations = df_finding_train[df_finding_train.image_id == img_id]
            #get information the bboxes (locations + labels)
            boxes_viz = img_annotations[["x_min", "y_min", "x_max", "y_max"]].to_numpy().tolist()
            label_viz =  img_annotations["class_id"].to_numpy().tolist()

            # DRAW THE BBOXES BEFORE REMOVE
            img_before = img_array.copy()
            for box, label in zip(boxes_viz, label_viz):
                x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
                color = label2color[label]
                img_before = draw_bbox(img = img_before,box =  list(np.int_(box)), 
                                       label = GetKey(label, dct_class_ids), color = color, 
                                       thickness = thickness)
            
            viz_imgs.append(img_before)

            # prepare input data of NMS function 
            boxes_list = []
            scores_list = []
            labels_list = []
            weights = []

            # have a list to save a single boxes in each image cause the NMS function do not allow this case
            boxes_single = []
            labels_single = []

            cls_ids = img_annotations["class_id"].unique().tolist()
            count_dict = Counter(img_annotations["class_id"].tolist()) # count the cls_id 

            for cid in cls_ids: 
                # check if have any the class_id of this img = 1 (mean just 1 bboxes in this img)
                if count_dict[cid] == 1: # --> tức là class_id đó chỉ có 1 bbox trên image
                    labels_single.append(cid)
                    boxes_single.append(img_annotations[img_annotations.class_id == cid][["x_min", "y_min", "x_max", "y_max"]].to_numpy().squeeze().tolist())
                else:
                    cls_list = img_annotations[img_annotations.class_id == cid]["class_id"].tolist()
                    labels_list.append(cls_list)
                    bboxs = img_annotations[img_annotations.class_id == cid][["x_min", "y_min", "x_max", "y_max"]].to_numpy()

                    '''Normalize the x_min, y_min, x_max, y_max
                    i.e: 
                    bboxs (before) = [[230. 458. 551. 610.]
                                      [230. 458. 552. 599.]
                                      [229. 437. 555. 587.]]
                    
                    bboxs (after) = [[0.33189033 0.58868895 0.7950938  0.7840617 ]
                                     [0.33189033 0.58868895 0.7965368  0.76992288]
                                     [0.33044733 0.56169666 0.8008658  0.75449871]]
                    '''
                    bboxs = bboxs /(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
                    
                    # np.clip --> Clip (limit) the values in an array.
                    # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
                    bboxs = np.clip(bboxs, 0, # minvalue
                                    1) # maxvalue 
                    '''
                    result of each loop:
                    i.e: only 1 label in this images
                    boxes_list = 
                    [[[0.3318903318903319, 0.5886889460154242, 0.7950937950937951, 0.7840616966580977], [0.3318903318903319, 0.5886889460154242, 0.7965367965367965, 0.7699228791773779], [0.33044733044733043, 0.5616966580976864, 0.8008658008658008, 0.7544987146529563]]]
                    scores_list = [[1.0, 1.0, 1.0]]
                    weights = [1]
                    '''
                    boxes_list.append(bboxs.tolist())
                    # np.ones(với shape là 1 array 1 chiều với length = length of cls_list)
                    scores_list.append(np.ones(len(cls_list)).tolist())
                    # you can ignore the weights cause this code just run single model's prediction 
                    weights.append(1)
            
            ### REMOVE
            if method == 1:
                final_boxes, final_scores, final_box_labels = ensemble_boxes_NMS.soft_nms(boxes=  boxes_list, 
                                                     scores = scores_list, 
                                                     labels = labels_list, 
                                                     weights = weights,
                                                     iou_thr = iou_thr,
                                                     sigma= sigma, 
                                                     thresh= skip_box_thr, 
                                                     method= 1)
            elif method ==2:
                final_boxes, final_scores, final_box_labels = ensemble_boxes_NMS.soft_nms(boxes=  boxes_list, 
                                                     scores = scores_list, 
                                                     labels = labels_list, 
                                                     weights = weights,
                                                     iou_thr = iou_thr,
                                                     sigma= sigma, 
                                                     thresh= skip_box_thr, 
                                                     method= 2)
            else: # method == 3
                final_boxes, final_scores, final_box_labels = ensemble_boxes_NMS.nms(boxes=  boxes_list, 
                                                     scores = scores_list, 
                                                     labels = labels_list, 
                                                     weights = weights,
                                                     iou_thr = iou_thr)
            
            # final_boxes
            # rescale the coordinates of boxes
        
            final_boxes = final_boxes*(img_array.shape[1], #x_min * width
                                       img_array.shape[0], #y_min * height
                                       img_array.shape[1], #x_max * width
                                       img_array.shape[0]) #y_max * height
            
            final_boxes = final_boxes.round(1) # làm tròn các số trong array (lấy 1 số đầu tiên sau dấu chấm động)
            final_boxes = final_boxes.tolist()
            final_boxes.extend(boxes_single)
            
            # final_box_labels
            final_box_labels = final_box_labels.astype(int).tolist()
            final_box_labels.extend(labels_single)

            print(f"loop = {i+1}")
            print(f"Labels before nms:\n{label_viz}\nand length is {len(label_viz)}")
            print(f"Labels after nms:\n{final_box_labels}\nand length is {len(final_box_labels)}")
            print(f"Bboxes location before nms:\n{boxes_viz}")
            print(f"Bboxes location after nms:\n{final_boxes}")

            # DRAWN THE BBOXES AFTER NMS
            img_after = img_array.copy()
            for box, label in zip(final_boxes, final_box_labels):
                color = label2color[label]
                img_after = draw_bbox(img_after, 
                                      list(np.int_(box)), 
                                      GetKey(label, dct_class_ids), 
                                      color, 
                                      thickness = thickness)
            
            viz_imgs.append(img_after)
            #print(f"End {i} loop")
            print()
        
        plot_multi_imgs(viz_imgs, cols = 2, cmap= None, 
                        name_fig= os.path.join(Charts_Images_dir,fig_save), close = True)

    # MODE = 2 REMOVE AND SAVE (ENTIRE IMAGES)
    else: # mode = 2
        id_lst = []
        label_boxes_after_NMS = []
        location_boxes_after_NMS = []
        
        # some case, in the image jus has single label for peer classes of this image
        '''i.e mean image_1 --> have 3 different classes: class_1, class_2, class_3
        and it's labeled only one time for each class 
        --> number_of_class_1 = number_of_class_2 = number_of_class_3 = 1 --> can't be used NMS 
        --> must skip this case 
        '''
        single_label_peer_class = 0

        for i, path in tqdm(enumerate(imgs_path[:]), total = len(imgs_path[:])):
            img_array  = cv2.imread(path)
            image_basename = Path(path).stem
            print(f"(\'{image_basename}\', \'{path}\')")
            id_lst.append(image_basename)
            img_annotations = df_finding_train[df_finding_train.image_id==image_basename]

            boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
            labels_viz = img_annotations['class_id'].to_numpy().tolist()
            print("Bboxes before nms:\n", boxes_viz)
            print("Labels before nms:\n", labels_viz)
            boxes_list = []
            scores_list = []
            labels_list = []
            weights = []
            
            boxes_single = []
            labels_single = []
            cls_ids = img_annotations['class_id'].unique().tolist()
            count_dict = Counter(img_annotations['class_id'].tolist())
            print(f"number of each class in this image = \n{count_dict}")

            for cid in cls_ids:
                ## Performing Fusing operation only for multiple bboxes with the same label
                if count_dict[cid]==1: 
                    labels_single.append(cid)
                    boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())
                else:
                    cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
                    labels_list.append(cls_list)
                    bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
                    
                    '''Normalize the x_min, y_min, x_max, y_max
                    i.e: 
                    bboxs (before) = [[230. 458. 551. 610.]
                                      [230. 458. 552. 599.]
                                      [229. 437. 555. 587.]]
                    
                    bboxs (after) = [[0.33189033 0.58868895 0.7950938  0.7840617 ]
                                     [0.33189033 0.58868895 0.7965368  0.76992288]
                                     [0.33044733 0.56169666 0.8008658  0.75449871]]
                    '''
                    bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0]) 
                    
                    # np.clip --> Clip (limit) the values in an array.
                    # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
                    bbox = np.clip(bbox, 0, 1) 
                    
                    boxes_list.append(bbox.tolist())
                    scores_list.append(np.ones(len(cls_list)).tolist())
                    
                    # you can ignore the weights cause this code just run single model's prediction 
                    weights.append(1)
            
            # check the special case
            if boxes_list ==  []:
                box_labels = None
                boxes = None
            
            else:
                if method == 1:
                    boxes, scores, box_labels = ensemble_boxes_NMS.soft_nms(boxes = boxes_list, 
                                                                            scores = scores_list, 
                                                                            labels = labels_list, 
                                                                            weights = weights,
                                                                            iou_thr = iou_thr,
                                                                            sigma= sigma, 
                                                                            thresh= skip_box_thr, 
                                                                            method= 1)
                elif method ==2:
                    boxes, scores, box_labels = ensemble_boxes_NMS.soft_nms(boxes = boxes_list, 
                                                                            scores = scores_list, 
                                                                            labels = labels_list, 
                                                                            weights = weights,
                                                                            iou_thr = iou_thr,
                                                                            sigma= sigma, 
                                                                            thresh= skip_box_thr, 
                                                                            method= 2)
                else: # method == 3
                    boxes, scores, box_labels = ensemble_boxes_NMS.nms(boxes = boxes_list, 
                                                                       scores = scores_list, 
                                                                       labels = labels_list, 
                                                                       weights = weights,
                                                                       iou_thr = iou_thr)
            
                # rescale
                boxes = boxes*(img_array.shape[1], img_array.shape[0], 
                img_array.shape[1], img_array.shape[0])
                boxes = boxes.round(1).tolist()
                boxes.extend(boxes_single)

                box_labels = box_labels.astype(int).tolist()
                box_labels.extend(labels_single)

            
            if box_labels == None:
                # SPECIAL CASE
                print("This image just has single label for peer classes in entire its classes")
            else:
                # NORMAL CASE --> NMS run
                print("Bboxes after nms:\n", boxes)
                print("Labels after nms:\n", box_labels)
                print("length of Bboxes after nms:\n", len(boxes))
                print("Length of Labels after nms:\n", len(box_labels))
            
            if box_labels == None:
                label_boxes_after_NMS.append(labels_viz)
                location_boxes_after_NMS.append(boxes_viz)
                single_label_peer_class +=1
            else:
                label_boxes_after_NMS.append(box_labels)
                location_boxes_after_NMS.append(boxes)
            
            print(f"End {i} loop")
        
        print(f"Entire the process has {single_label_peer_class} special case")
        # CONVERT TO CSV FILE 
        lst_rows = []
        invalid = 0

        for loop in range(len(label_boxes_after_NMS)):
            for i, class_id in enumerate(label_boxes_after_NMS[loop]):
                # Check if the location_boxes_after_NMS[loop] is the special case or not
                '''i.e:
                location_boxes_after_NMS[loop] = [1,2,3,4]
                len(np.array(location_boxes_after_NMS[loop]).shape)
                >>> = 1
                '''
                if len(np.array(location_boxes_after_NMS[loop]).shape) == 1:
                    '''i.e:
                    location_boxes_after_NMS[loop] = [1,2,3,4] (valid syntax)
                    len(location_boxes_after_NMS[loop]) == 4
                    '''
                    if len(location_boxes_after_NMS[loop]) ==  4: # (valid syntax)
                        lst_row = [id_lst[loop],
                                   class_id, 
                                   location_boxes_after_NMS[loop][0], 
                                   location_boxes_after_NMS[loop][1],
                                   location_boxes_after_NMS[loop][2],
                                   location_boxes_after_NMS[loop][3]]
                    else: # (invalid syntax)
                        invalid +=1
                        continue
                
                else:
                    '''i.e:
                    location_boxes_after_NMS[loop] = [[1,2,3,4], [1,2,3,4]]
                    len(np.array(location_boxes_after_NMS[loop]).shape)
                    >>> = 2 
                    '''
                    lst_row = [id_lst[loop],
                               class_id, 
                               location_boxes_after_NMS[loop][i][0], 
                               location_boxes_after_NMS[loop][i][1],
                               location_boxes_after_NMS[loop][i][2],
                               location_boxes_after_NMS[loop][i][3]]
                
                lst_rows.append(lst_row)
        
        #CREATE CSV_FILE AND SAVE 
        df_NMS = pd.DataFrame(data = lst_rows, columns = ["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"])
        df_NMS.to_csv(os.path.join(csv_txt_json_data, file_save))

### RUN CODE
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', help= 'cvs_file must contatin image_path, x_min, y_min, x_max, y_max, class_id', type= str,
                        default= 'dataset_after_processing_fe.csv')
    parser.add_argument('--mode', help= 'mode = 1 (default), Remove and display image; mode = 2, Remove and save', type= int, default= 1)
    parser.add_argument('--method', help='method = 3 (default) --> NMS; method = 1 --> soft-NMS with linear penalty; '
                                         'method = 2 --> soft-NMS with Gaussian penalty', type = int, default= 3)
    parser.add_argument('--start', help= 'start index of image_path', type = int, default= 1)
    parser.add_argument('--stop', help='stop index of image_path', type = int, default= 5)
    parser.add_argument('--fig_save', help = "run if mode = 1, figure's name", type= str, default= 'result_NMS.png')
    parser.add_argument('--file_save', help = "run if mode = 2, name of csv file", type= str, default= 'after_remove.csv')
    parser.add_argument('--iou_thr', help="default = 0.5; iou threshold", type = float, default= 0.5)
    parser.add_argument('--skip_box_thr', help="default = 0.001; threshold for boxes to keep (important for SoftNMS)", type = float,
                        default= 0.001)
    parser.add_argument('--sigma', help="default = 0.1; Sigma value for SoftNMS", type = float, default= 0.1)


    args = parser.parse_args()
    main(path_csv = args.csv_file, mode = args.mode, method= args.method, start = args.start,
         stop = args.stop, fig_save = args.fig_save, file_save= args.file_save, sigma= args.sigma,
         iou_thr= args.iou_thr, skip_box_thr= args.skip_box_thr)


    


            

        





                        
                
    
                




