""""
Miscellaneous function to run OpenCV DNN YoloV5
"""
import cv2 as cv
import time
import sys
import numpy as np


class prepDetect:
    """
    Variables that will have an initial values that will be updated will be created in the class"""

    frame=None 
    conf_thresh, NMS_THRESHOLD,score_thresh=0.4, 0.4,0.25    

    def __init__(self, class_path, weight_onnx, INPUT_WIDTH, INPUT_HEIGHT):
        
        self.class_path=class_path #as .txt file
        self.vid_path=None
        self.weight_onnx=weight_onnx #path to weight
        self.INPUT_WIDTH, self.INPUT_HEIGHT = INPUT_WIDTH, INPUT_HEIGHT 
        self.outs=None        
        self.INPUT_WIDTH, self.INPUT_HEIGHT= 640,640
        self.conf_thresh, self.NMS_THRESHOLD,self.score_thresh 
        #print(f'input w, h {self.INPUT_WIDTH, self.INPUT_HEIGHT}')
        
        

    ###Dataset##
    def load_capture(self):        
        capture = cv.VideoCapture(self.vid_path)
        return capture


    def formatFrame(self, frame):
        """ Creates a black square canvas around the frame"""
        prepDetect.frame=frame
        
        row, col, _ = prepDetect.frame.shape
        _max = max(col, row)
        self.frame_reshaped = np.zeros((_max, _max, 3), np.uint8)
        self.frame_reshaped[0:row, 0:col] = prepDetect.frame
        self.frame_reshaped[0:row, 0:col] = prepDetect.frame
        #print(f'resized frame shape check', self.frame_reshaped.shape)
        return self.frame_reshaped



    #####Model

    def load_classes(self):
        self.class_list = []
        with open(self.class_path, "r") as f: 
            self.class_list = [cname.strip() for cname in f.readlines()]
        return self.class_list


    def model(self):
        """
        Builds model once
        """
        
        net = cv.dnn.readNet(self.weight_onnx)
        print("Running on CPU")
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)        
        return net


    def detect(self, image, net):#(image, net):
        """ Calls predict on each frame
        image is likely the resized_reshaped image"""       
        
        blob = cv.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds



     #######NMS

    def detectNMS(self, outs, image, conf_thresh, score_thresh, nms_threshold):         
        """"
        image is likely the resized_reshaped image
        """
        self.outs=outs
        prepDetect.conf_thresh, prepDetect.NMS_THRESHOLD,prepDetect.score_thresh=conf_thresh, nms_threshold, score_thresh
        #print(f'cns: {prepDetect.conf_thresh, prepDetect.NMS_THRESHOLD,prepDetect.score_thresh}')
        class_ids = []
        confidences = []
        boxes = []
        #print('nms outs', len(outs))

        rows = self.outs.shape[0]

        image_width, image_height, _ = image.shape

        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = self.outs[r]
            confidence = row[4]
            if confidence >= prepDetect.conf_thresh:
                
                classes_scores = row[5:]
                _, _, _, max_indx = cv.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > prepDetect.score_thresh):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        #print('nms boxes', boxes)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, prepDetect.score_thresh, prepDetect.NMS_THRESHOLD) 
        #print(f'nms idx {indexes}')
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
        
        

        return result_class_ids, result_confidences, result_boxes




##################Video Capture

import cv2, queue, threading, time
import logging
from logging.handlers import RotatingFileHandler
from logging.config import fileConfig


class VideoCapture:
  r"""
  This custom function creates a separate thread that caches
  and clears the 'internal' buffer retained
  by the cv2.VideoCapture() object.
  The buffering creates about 30 seconds
  lag in between frames being analyzed and imshow() plot.
  Implementing this function solves that problem. 
    """

  def __init__(self, name):    
    self.name=name    
    self.cap = cv2.VideoCapture(self.name) 
     
    self.q = None
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    log_file='camera_status.log'
    fileConfig('./persondetectapp/src/model_data/config/config_cam_logfile.config')
    logger = logging.getLogger(__name__)

    ##################
    open_time = time.time()
    self.cap.open(self.name) 
    logger.warning('1_cap.open() duration:   {}'.format(time.time()-open_time))    
    ##################
      
    while True:
            
      if not self.cap.isOpened():
        self.cap.release()
        time.sleep(2)
        ##################
        open_time = time.time()
        self.cap.open(self.name) 
        logger.warning('2_cap.open() duration:   {}'.format(time.time()-open_time))
        continue
              
      else:
        vid_time = time.time()
        ret, frame = self.cap.read()    
        
        if not ret:
          self.cap.open(self.name)
          #break
          continue
        self.q = frame
        logger.warning("1_video capture_empty queue time.  {}".format(time.time()-vid_time))  
        #print(f'Is {self.name} camera open? : {self.cap.isOpened()}')    
     
  def read(self):
    return self.q#.get()

  # Returns the height and width of camera frame.
  def framesize(self):
    return [self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)]




#raw points are stored as xi,yi sequence and comma delimited not in tuples that we need.
def coordsFormatter(path2=''):
    """
    This functions seeks to convert. 
    example raw format is in string format:
    [6,52.29,498.24,178.577,450.882,304.863] where first item is drawing style(polygon, point etc).
    The rest is x,y,x,y,x,y coordinates.
    This function:
    Watches the working directory
    Asserts the csv file exists in the right directory. Needs a single file in the dir.
    Excludes the first and last two characters, then splits the string to make them eligible for numerical formating
    Appends each list into a container
    Subset each as tuple(x,y)
    returns a list of list of coordinates as tuple(x,y)

    """
    import os
    import glob     
    import csv
    data_path=glob.glob(os.path.join(path2,'*.csv'))
    print('filename', data_path[0])
    
    assert os.path.exists(data_path[0]), 'Needs the redzone region .csv file inside /rz_coords directory?'
    
    with open(data_path[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        coords_all=[]
        print('reader', reader)
        for idx, row in enumerate(reader):
            if idx>8:
                str_coords = list(row.values())[1][-2] #subset for list of coords
                listx=[float(x) for x in str_coords[1:-2].split(',')[1:]]
                coords_all.append(listx)

    real_coords=[]
    for j in coords_all:
        coords = [j[i:i + 2] for i in range(0, len(j), 2)]
        real_coords.append(coords)
        print('real coords', real_coords)
    return real_coords





    