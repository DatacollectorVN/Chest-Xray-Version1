# copy from https://github.com/ZFTurbo/Weighted-Boxes-Fusion

import numpy as np
from numba import jit
#https://numba.pydata.org/numba-doc/latest/user/jit.html

'''CHECK AND PREAPARE THE BOXES
INPUT:
+ boxes: array of all boxes that contain the coordiantes of boxes (x_min, y_min, x_max, y_max) with normalize range from 0 to 1
+ scores: array of all scores of each boxes
+ labels: array of all labels of each boxes

GOALS:
+ check boxes with invalid format (<0; > 1, area = 0)
+ convert the new standard format (x_min, y_min, x_max, y_max)
OUTPUT:
return boxes, scores, labels (after validated)
'''
def check_prepare_boxes(boxes, scores, labels):
    '''
    :param boxes: array of all boxes that contain the coordiantes of boxes (x_min, y_min, x_max, y_max) 
        with normalize range from 0 to 1
    :param scores: array of all scores of each boxes
    :param labels: array of all labels of each boxes

    :return: boxes, scores, labels (after validated)
    '''
    boxes_output = boxes.copy()

    # CHECK IF THE VALUE IN BOXES IS NEGATIVE OR NOT 
    
    # return False value if >0 and True if <0 
    # i.e: [[1,2,-4]] --> return: [[False, False, True]] 
    check = (boxes_output < 0)

    #check.astype(np.int64) --> True = 1, False = 0
    #  [[1,2,-4]] --> [[False, False, True]] --> [[0, 0, 1]]
    # .sum(): [[0, 0, 1]] = 1
    check_sum = check.astype(np.int64).sum()

    # if check_sum > 0 --> mean: exist at least 1 value in the box that is negative
    if check_sum > 0: 
        print(f"WARING: FIXED {check_sum} BOXES COORDINATES < 0")
        print("DEFAULT: ASSIGNED ITS VALUE = 0")
        
        # assign the values that are negative equal 0 
        # i.e: [[1,2,-4]] --> [[1,2,0]]
        boxes_output[check] = 0
    
    # CHECK IF THE VALUE IN BOXES IS OVER 1 OR NOT 
    # cause we must to normalize all of value of box's coordinates in range [0,1]
    
    check = (boxes_output > 1)
    check_sum = check.astype(np.int64).sum()
    
    if check_sum > 0: 
        print(f"WARING: FIXED {check_sum} BOXES COORDINATES > 0\nCheck that your boxes was normalized at [0, 1]")
        print("DEFAULT: ASSIGNED ITS VALUE = 1")
        
        # assign the values that are over than 1 
        # i.e: [[1,2,0.5]] --> [[1,1,0.5]]
        boxes_output[check] = 1
    
    # CHECK THE FORMAT OF BOXES (x_min, y_min, x_max, y_max)
    # but in all of boxes, some box have invalid format --> we must to convert it 
    boxes_copy = boxes_output.copy()
    # x_min
    '''
    np.array([[5,5,10,10],
                [4,5,9,10], 
                [0,0,0,0]])
    np.min(lst[:, [0, 2]], axis=1)
    >>> array([5, 4, 0])
    '''
    boxes_output[:, 0] = np.min(boxes_copy[:, [0, 2]], axis = 1)

    # x_max 
    '''
    np.array([[5,5,10,10],
                [4,5,9,10], 
                [0,0,0,0]])
    np.max(lst[:, [0, 2]], axis=1)
    >>> array([10,  9,  0])
    '''
    boxes_output[:, 2] = np.max(boxes_copy[:, [0, 2]], axis = 1)

    #y_min
    '''
    np.array([[5,5,10,10],
                [4,5,9,10], 
                [0,0,0,0]])
    np.min(lst[:, [1, 3]], axis=1)
    >>> array([5, 5, 0])
    '''
    boxes_output[:, 1] = np.min(boxes_copy[:, [1, 3]], axis = 1)

    #y_max
    '''
    np.array([[5,5,10,10],
                [4,5,9,10], 
                [0,0,0,0]])
    np.max(lst[:, [1, 3]], axis=1)
    >>> array([10, 10, 0])
    '''
    boxes_output[:, 3] = np.max(boxes_copy[:, [1, 3]], axis = 1)

    # CHECK AREA OF EACH BOX --> IF AREA = 0 --> REMOVE
    area = (boxes_output[:, 2] - boxes_output[:, 0]) * (boxes_output[:, 3] - boxes_output[:, 1])
    check = (area == 0)
    check_sum = check.astype(np.int64).sum()
    if check_sum >0:
        print(f"WARING. REMOVE {check_sum} BOXES WITH ZERO AREA")

        '''
        lst = array([[ 5,  5, 10, 10],
               [ 4,  5,  9, 10],
               [ 0,  0,  0,  0]])
        --> area = array([25, 25,  0])
        --> area > 0 = array([True, True, False])
        --> lst[area > 0] = array([[ 5,  5, 10, 10],
                                   [ 4,  5,  9, 10]])
        '''
        boxes_output = boxes_output[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]

    return boxes_output, scores, labels


''' Soft_NMS_function:
INPUT: 
+ boxes_by_label: array of coordinates of boxes with single label
+ scores_by_label: array of scores of boxes with single label
+ iou_thr
+ sigma: parameter for soft-NMS
+ thresh: parameter for soft-NMS, is the threshold for scores (after penalized by soft-NMS)
+ method: configure method to calculate: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS

OUTPUT: --> list of index of boxes that we need to keep (for single label)

'''
def soft_NMS_function(boxes_by_label, scores_by_label, iou_thr, sigma, thresh, method):
    '''
    :param boxes_by_label: boxes format [x_min, y_min, x_max, y_max]
    :param scores_by_label: scores for boxes
    :param iou_thr: required iou 
    :param sigma:  
    :param thresh: 
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    
    :return: index of boxes to keep
    '''
    num_boxes = boxes_by_label.shape[0]

    # i.e num_boxes = 5 
    # >>> indexes = array([0, 1, 2, 3, 4])
    indexes = np.array([np.arange(num_boxes)])

    # indexes concatenate boxes with the last columns
    # i.e: 
    # >>> output: array([[0.1 , 0.31, 0.71, 0.61, 0.  ],
    #                   [0.02, 0.53, 0.11, 0.94, 1.  ],
    #                   [0.03, 0.24, 0.12, 0.35, 2.  ]])
    boxes_by_label = np.concatenate([boxes_by_label, indexes.T], axis = 1)
    boxes =  boxes_by_label
    
    # get all value of individual coordinate of boxes
    x_min = boxes[:, 0] 
    y_min = boxes[:, 1] 
    x_max = boxes[:, 2] 
    y_max = boxes[:, 3] 

    scores =  scores_by_label
    areas = (x_max - x_min) * (y_max - y_min)

    
    for i in range(num_boxes):
        # intermediate parameters for later parameters exchange
        t_BD = boxes[i, :].copy()
        t_score = scores[i].copy()
        t_area = areas[i].copy()
        pos = i+1

        if i != num_boxes-1:
            # get maximum value of scores arrays with starting after the i-th index of scores array
            max_score = np.max(scores[pos:])
            # get the indexes of array with maximum value
            max_pos = np.argmax(scores[pos:])
        else:
            # when i == num_boxes -1 mean --> the last index
            # max_scores = the last of scores list
            max_score = scores[-1]
            max_pos = 0

        # we want to reorder the value of boxes, scores and areas according the the descending corresponding 
        # score
        if t_score < max_score:
            # we should to swap the position 
            
            # SWAP THE BOXES ARRAY
            boxes[i, :] = boxes[max_pos + pos, :]
            boxes[max_pos + pos, :] = t_BD
            # reset value t_BD
            t_BD = boxes[i, :]

            #SWAP THE SCORES ARRAY
            scores[i] = scores[max_pos + pos]
            scores[max_pos + pos] = t_score
            # reset value t_score
            t_score = scores[i]

            #SWAP THE AREAS ARRAY
            areas[i] = areas[max_pos + pos]
            areas[max_pos + pos] = t_area
            # reset value t_area
            t_area = areas[i]
        
        # COMPUTE IOU

        # np.maximum() -->  https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        # elemenwise https://tek4.vn/element-wise-lap-trinh-neural-network-voi-pytorch-bai-11/
        '''
        i.e: 
        boxes[i, 1] = 0.53
        boxes[pos:, 1] = [0.24, 0.56, 0.33, 0.66]
        np.maximum(boxes[i, 1], boxes[pos:, 1])
        >>> [0.53, 0.56, 0.53, 0.66]
        '''

        # boxes[i, 1] --> the value of coordinate i-th of y_min
        # and the reason why we use y_min for x_min, otherwise --> see the video meshgrid: https://www.youtube.com/watch?v=qQeKEawn_HI
        xx_min = np.maximum(boxes[i, 1], boxes[pos:, 1])
        yy_min = np.maximum(boxes[i, 0], boxes[pos:, 0])

        xx_max = np.maximum(boxes[i, 3], boxes[pos:, 3])
        yy_max = np.maximum(boxes[i, 2], boxes[pos:, 2])

        w = np.maximum(0.0, xx_max - xx_min)
        h = np.maximum(0.0, yy_max - yy_min)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)


        # CONFIGURE METHOD
        if method == 1: # use linear
            weights = np.ones(iou.shape)
            weights[iou > iou_thr] = weights[iou > iou_thr] - iou[iou > iou_thr]

        elif method == 2: # gaussian
            weights = np.exp(-(iou * iou) / sigma)

        else: # NMS
            weights = np.ones(iou.shape)


            weights[iou > iou_thr] = 0

        # Penalize the scores with the corresponding weight
        scores[pos:] = weights * scores[pos:]

    # select the boxes and keep the corresponding indexes
    # [scores > thresh] --> return array keeping the value True (scores > thresh) or False (scores <= thresh)
    # boxes[:, 4][scores > thresh] --> return the boxes value with indexes that have True
    inds = boxes[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


'''NMS Function
INPUT: 
+ boxes_by_label: array of coordinates of boxes with single label
+ scores_by_label: array of scores of boxes with single label
+ iou_thr: IOU value for boxes

OUTPUT: --> list of index of boxes that we need to keep (for single label)
'''

# Using this decorator, you can mark a function for optimization by Numba’s JIT compiler
# https://numba.pydata.org/numba-doc/latest/user/jit.html

@jit(nopython = True)
def NMS_function(boxes_by_label, scores_by_label, iou_thr):
    '''
    :param boxes_by_label: array of coordinates of boxes with single label
                           float coordinates on range [0; 1]
    :param scores_by_label: array of scores of boxes with single label
    :param iou_thr: IOU value for boxes

    :return: list of index of boxes that we need to keep (for single label)
    '''
    boxes = boxes_by_label
    scores = scores_by_label

    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    areas = (x_max - x_min) * (y_max - y_min)
    '''i.e: scores = [0.6       , 0.53333333, 0.13333333, 0.26666667, 0.46666667, 0.55      ]
    scores.argsort() 
    >>> [2, 3, 4, 1, 5, 0] --> indices with ascending order 
    scores.argsort()[::-1] --> indexces with descending order
    '''
    # see more to know https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
    des_order = scores.argsort()[::-1] # Returns the indices descending order

    keep = []
    while des_order.size > 0:
        i = des_order[0]
        keep.append(i)

        # IOU calculate
        ### NOTE: des_order[1:] --> return array x[<array>] --> error BUT when we use @jit(nopython = True)
        # --> Don't have error
        '''i.e: @jit(nopython = True)
        THE FIRST WHITE LOOP
        x_min = [0.1 , 0.02, 0.03, 0.04, 0.12, 0.38]
        des_order = [0, 1, 4, 6, 3, 7, 5, 2, 8] --> des_order[1:] = [1, 4, 6, 3, 7, 5, 2, 8]
        i = 0 
        x_min[i]
        >>> 0.1
        x_min[des_order[1:]]
        >>> [2.00000000e-002, 1.20000000e-001, 4.94065646e-324, 4.00000000e-002,
             6.95281955e-310, 3.80000000e-001, 3.00000000e-002, 0.00000000e+000]
        2.00000000e-002 = 0.02 = x_min[des_order[1:][0]] = x_min[1]
        4.94065646e-324 --> generate randomly from compiler cause x_min[des_order[1:][2]] = x_min[6] --> out of indexes
        
        np.maximum(x_min[i], x_min[des_order[1:]]) see more examples above 
        >>> [0.1 , 0.12, 0.1 , 0.1 , 0.1 , 0.38, 0.1 , 0.1 ]
        '''

        xx_min = np.maximum(x_min[i], x_min[des_order[1:]])
        yy_min = np.maximum(y_min[i], y_min[des_order[1:]])
        xx_max = np.minimum(x_max[i], x_max[des_order[1:]])
        yy_max = np.minimum(y_max[i], y_max[des_order[1:]])

        '''i.e 
        THE FRIST LOOP
        w = [0.01, 0.59, 0.  , 0.61, 0.  , 0.33, 0.02, 0.  ]
        h = [0.08, 0.28, 0.  , 0.05, 0.  , 0.  , 0.04, 0.  ]
        iou = [0.0036513 , 0.81059863, 0., 0.0692395, 0., 0., 0.0041645, 0.]
        iou_thr = 0.5 
        
        np.where(iou <= iou_thr) # don't error when use @jit(nopython = True)
        >>> (array([0, 2, 3, 4, 5, 6, 7], dtype=int64),)
        inds = np.where(iou <= iou_thr)[0] 
        >>> array([0, 2, 3, 4, 5, 6, 7] --> mean keep the indices with iou < iou_thr         
        des_order
        >>> [0, 1, 4, 6, 3, 7, 5, 2, 8] 
        des_order = des_order[inds + 1]
        >>> array([1, 6, 3, 7, 5, 2, 8], dtype=int64)
        
        + 1 = des_order[inds + 1] = des_order[0 + 1] = des_order[1]
        + 6 = des_order[inds + 1] = des_order[2 + 1] = des_order[3]
        '''
        w = np.maximum(0.0, xx_max - xx_min)
        h = np.maximum(0.0, yy_max - yy_min)
        intersection = w*h
        iou = intersection / (areas[i] + areas[des_order[1:]] - intersection)
        inds = np.where(iou <= iou_thr)[0]
        # update des_order
        des_order = des_order[inds + 1]

    return keep


'''NMS Configuration: to control use NMS or Soft-NMS 
INPUT: 
+ boxes: list of boxes prediction from each model. It has shape (N_models, 1)
i.e boxes= [[
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61]
        ],[
        [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        ]]
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]

+ scores: list of scores for each model
i.e scores = scores_list = [[0.9, 0.8], [0.5, 0.8]] 

+ labels: list of labels for each model
i.e labels = [[0, 1], [1, 1]]

+ method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
+ sigma: Sigma value for SoftNMS
+ thresh: threshold for boxes to keep (important for SoftNMS)
+ weights: list of weights for each model. Default: None, which means weight == 1 for each model

OUTPUT:
+ boxes: array of boxes coordinates (Order of boxes: x1, y1, x2, y2). 
+ scores: array of confidence scores
+ labels: array of boxes labels
'''

def nms_config(boxes, scores, labels, method = 3, iou_thr = 0.5, weights = None,
                sigma = 0.5, thresh = 0.001):
    '''
    :param boxes = list of boxes prediction from each model. It has shape (N_models, 1)
        if using np.concatenate() --> (N_model, 4)
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]
    :param scores: list of scores for each model
    :param labels: list of labels for each model
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :param iou_thr: IoU value for boxes to be a match 
    :param sigma: Sigma value for SoftNMS
    :param thresh: threshold for boxes to keep (important for SoftNMS)
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each models
    
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels

    '''

    # if weights is existence
    if weights is not None:
        # check the condition
        if len(boxes) != len(weights):
            print(f"INCORRECCT NUMBER OF WEIGHTS: {len(weights)}. MUST BE: {len(boxes)}")
        
        else:
            # convert weight to array
            weights = np.array(weights)

            # multiple weights with scores with formula:
            '''
            i.e 
            scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
            weights = [2, 1]

            #AFTER RUN:
            scores_list = [array([0.6       , 0.53333333, 0.13333333, 0.26666667, 0.46666667]),
                           array([0.16666667, 0.26666667, 0.23333333, 0.1       ])]
            >>> 0.6 = 0.9 * 2 / (2+1)
            '''
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()
    
    # concatenate everything inside boxes, scores, labels
    '''
    boxes_list = [[
    [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
    ],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
    ]]

    scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
    labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]

    # AFTER RUN
    >>> boxes_list = 
    array([[0.  , 0.51, 0.81, 0.91],
       [0.1 , 0.31, 0.71, 0.61],
       [0.01, 0.32, 0.83, 0.93],
       [0.02, 0.53, 0.11, 0.94],
       [0.03, 0.24, 0.12, 0.35],
       [0.04, 0.56, 0.84, 0.92],
       [0.12, 0.33, 0.72, 0.64],
       [0.38, 0.66, 0.79, 0.95],
       [0.08, 0.49, 0.21, 0.89]])
    
    >>> scores_list = 
    array([0.9, 0.8, 0.2, 0.4, 0.7, 0.5, 0.8, 0.7, 0.3])

    >>> label_list = 
    array([0, 1, 0, 1, 1,1, 1, 1, 0])
    '''
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # check, fix the coordinates and remove zero areas
    boxes, scores, labels = check_prepare_boxes(boxes, scores, labels)

    
    # RUN NMS INDEPENTLY FOR EACH LABEL
    unique_labels = np.unique(labels) 
    final_boxes = []
    final_scores = []
    final_labels = []

    for l in unique_labels:
        # return array with coorespond True or Flase that's satisfied by the condtion
        # i.e:        
        # labels = array([0, 1, 0, 1, 1,1, 1, 1, 0])
        # l = 1; check = (labels == l)
        # >>> Return array([False,  True, False,  True,  True,  True,  True,  True, False])
        check = (labels == l)
        
        # boxes_by_label and scores_by_label --> return the value with True index
        # boxes_by_label and scores_by_label will contain all infomation according to specific l (i.e: l = 1)
        boxes_by_label = boxes[check]
        scores_by_label = scores[check]
        labels_by_label = np.array([l] * len(boxes_by_label))

        if method !=3: # mean use soft-NMS
            # call soft-NMS function
            # >>> return the positions that we need to keep
            keep = soft_NMS_function(boxes_by_label= boxes_by_label.copy(),
                                     scores_by_label= scores_by_label.copy(),
                                     iou_thr= iou_thr,
                                     sigma= sigma,
                                     thresh= thresh,
                                     method= method)
        
        else: 
            # call NMS function
            # >>> return the positions that we need to keep
            # use faster function with jit(nopython = True)
            keep = NMS_function(boxes_by_label= boxes_by_label.copy(),
                                scores_by_label= scores_by_label.copy(),
                                iou_thr= iou_thr)


        final_boxes.append(boxes_by_label[keep])
        final_scores.append(scores_by_label[keep])
        final_labels.append(labels_by_label[keep])
    

    # after keep index of all labels --> concatenate
    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)

    return final_boxes, final_scores, final_labels

### FUNCTIONS ARES USED BY USERS

### 1. SHORT CALL NMS
'''nms
INPUT:
+ boxes: list of boxes prediction from each model. It has shape (N_models, 1)
i.e boxes= [[
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61]
        ],[
        [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        ]]
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]

+ scores: list of scores for each model
i.e scores = scores_list = [[0.9, 0.8], [0.5, 0.8]] 

+ labels: list of labels for each model
i.e labels = [[0, 1], [1, 1]]

OUTPUT:
+ boxes: array of boxes coordinates (Order of boxes: x1, y1, x2, y2). 
+ scores: array of confidence scores
+ labels: array of boxes labels
'''
def nms(boxes, scores, labels, iou_thr = 0.5, weights = None):
    '''
    :param boxes: list of boxes prediction from each model. It has shape (N_models, 1)
        if using np.concatenate() --> (N_model, 4)
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]
    :param scores: list of scores for each model
    :param labels: list of labels for each model
    :param iou_thr: IoU threshold value for boxes
    :param weights: list of weights for each model. Default: None, which means weight == 1 for all models

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    return nms_config(boxes = boxes, scores= scores, labels= labels, method= 3, iou_thr= iou_thr, weights= weights)

### 2. SHORT CALL SOFT_NMS
'''soft_nms
INPUT: 
+ boxes: list of boxes prediction from each model. It has shape (N_models, 1)
i.e boxes= [[
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61]
        ],[
        [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        ]]
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]

+ scores: list of scores for each model
i.e scores = scores_list = [[0.9, 0.8], [0.5, 0.8]] 

+ labels: list of labels for each model
i.e labels = [[0, 1], [1, 1]]

+ method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
+ sigma: Sigma value for SoftNMS
+ thresh: threshold for boxes to keep (important for SoftNMS)
+ weights: list of weights for each model. Default: None, which means weight == 1 for each model

OUTPUT:
+ boxes: array of boxes coordinates (Order of boxes: x1, y1, x2, y2). 
+ scores: array of confidence scores
+ labels: array of boxes labels
'''
def soft_nms(boxes, scores, labels, iou_thr = 0.5, weights = None,
             method = 2, thresh = 0.001, sigma = 0.5):
    '''
    :param boxes: list of boxes prediction from each model. It has shape (N_models, 1)
        if using np.concatenate() --> (N_model, 4)
        order of boxes: x_min, y_min, x_max, y_max (float) with normalized coordinates [0; 1]
    :param scores: list of scores for each model
    :param labels: list of labels for each model
    :param iou_thr: IoU threshold value for boxes
    :param weights: list of weights for each model. Default: None, which means weight == 1 for all models
    :param method: (importance) method = 1 - Linear penalty; method = 2 - Gaussian penalty
    :param thresh: threshold for boxes to keep (important for SoftNMS)
    :param sigma: Sigma value for SoftNMS

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    return nms_config(boxes= boxes, scores= scores, labels= labels, iou_thr= iou_thr, method= method,
                      weights= weights, sigma = sigma, thresh= thresh)


