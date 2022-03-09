# YOLO object detection
import configparser
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyModbusTCP.client import ModbusClient
from pyModbusTCP.server import ModbusServer, DataBank
import logging
from logging.handlers import RotatingFileHandler
from logging.config import fileConfig
from model_data.utils.utils_v5 import prepDetect, VideoCapture, coordsFormatter
import os
import glob 



config = configparser.ConfigParser()
config.read('./persondetectapp/src/model_data/config/config_live_cam.config') #reads in the config file that has inference params below

#Arguments values for livestream_detect.py inference file.
NMS_THRESHOLD=config.getfloat("MODELINFO", "nms_thres")
SCORE_THRESHOLD=config.getfloat("MODELINFO", "score_thres")
config_thresh=config.getfloat("MODELINFO", "conf_thres")




class detectCount(prepDetect):
    r"""
    Class for model data and the cpu-based prediction on camera streaming. 
    config_path = path to configuration file
    weights_path = path the pre-trained model weights
    class_path = path to class labels used for training
    img_size = image size compatible with the model
    conf_thres = confidence threshold for the prediction
    nms_thres = non-maximum suppression threshold for the model
    host = host ip address for the modbus protocol
    port = port address for the modbus protocol.
    """


    protocol, user, password, ip = None, None, None, None #will be updated below
    def __init__(self, class_path, weight_onnx, INPUT_WIDTH, INPUT_HEIGHT, host, port):
        super().__init__(class_path, weight_onnx, INPUT_WIDTH, INPUT_HEIGHT)

        self.host=host
        self.port=port
        #self.conf_thresh, self.NMS_THRESHOLD, self.SCORE_THRESHOLD=0.4, 0.4,0.25 

        t0 = time.time()

    ####Read Frames from Camera
    def stream_count(self, protocol='rtsp', user='root', password='admin', ip='10.202.109.131'): 
       
        """Get Coordinates"""
        rz_coords=None
        coords_dir='./persondetectapp_framework/rz_coords'
        #print('rz coords', os.listdir(coords_dir))
        try:
            rz_coords = coordsFormatter(path2=coords_dir) #takes a directory Expects a list type
            #print('rz_coords', rz_coords)
        except Exception as ex:
            logging.warning(f"Error in getting redzone coords from file: {ex!r}")
            print(f"Error in getting redzone coords from file: {ex!r}")

        
        
        
        """ Arg values obtained from config file"""
        detectCount.protocol, detectCount.user, detectCount.password, detectCount.ip = protocol, user, password, ip
        
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        fileConfig('./persondetectapp/src/model_data/config/config_cam_logfile.config')
        logger = logging.getLogger(__name__)

        coords_config=configparser.ConfigParser()
        coords_config.read('./persondetectapp/src/model_data/config/config_rz_coords.config')


        ##########Start ModBus#######################
        server = ModbusServer(host=self.host, port=self.port, no_block=True)
        client=ModbusClient(host=self.host, port=self.port)

        server.start()
        print(f'Modbus status: server: {server.is_run}, client: {client.open()}')

        ####################Create cap object for streaming frames#################
        ip_add = str('{}://{}:{}@{}/axis-media/media.amp'.format(detectCount.protocol, detectCount.user, detectCount.password, detectCount.ip))
        ip_add1 = 'rtsp://root:admin@10.201.222.35/axis-media/media.amp'
        
        net = detectCount.model(self)      
        #cap = VideoCapture(ip_add) #accessing the camera for reading 
        #cap = VideoCapture(ip_add) #
        #cap = VideoCapture(ip_add1) #
        cap = cv.VideoCapture('./persondetectapp/src/test_rig_img.png') #accessing the camera for reading 
        #print('cap', cap)
        
        frame_count = 0
        total_frames = 0
        fps = -1
        class_list = detectCount.load_classes(self)
        #while True: #(cap.isOpened()):
        start_time_here = time.time()
        t0=time.time()
        # if not server.is_run:
        #     logging.warning("Modbus is not running") 
        #     continue        
        start_time_2 = time.time()
        #time.sleep(1) #'sleep' for 1 second in between.
        _,frame = cap.read()
        #print(f'read time {time.time()-start_time_2}') 
                    

        ''' for videocapture'''
        # if frame is None: continue
        print(f'frame shape', frame.shape)

        inputImage = detectCount.formatFrame(self,frame)
        print(f'input frame {inputImage.shape}')

        ##Define the redzone geometry but don't quit if not found#####
        if not rz_coords: rz_coords = [(0,0),(frame.shape[1], frame.shape[0])] #if the error in getting coord from file use full frame.

        #####Detection step:  blobNMS returns boxes, indexes,classIDs,confidences
        t2=time.time()
        outs = detectCount.detect(self,image=inputImage, net=net)
        #print(f"detection outs {len(outs), outs}")
        #print(f'detection time {time.time()-t2}')
        class_ids, confidences, boxes = detectCount.detectNMS(self,outs=outs[0], image=inputImage, conf_thresh=config_thresh, score_thresh=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD)
        #print(f'detection to NMS {time.time()-t2}')
        print(f'boxes {boxes}')


        frame_count += 1
        total_frames += 1
        box_coords=[]
        class_label=[]
        people=0
        in_rz=None
        
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            print(f'boxes', box)
            bbox_lmidpt = np.array((box[0]+(box[2]/2),(box[1]+box[3]))).astype(np.float32) #x,y of lower midpoint of each bounding box
            #print('bbox1', bbox_lmidpt)


            if len (rz_coords)>1: #for list of lists
                #print(f'{len(rz_coords)} rz region defined')
                pts= np.array(rz_coords[0]).astype(np.float32).reshape((-1,1,2)) #ensure dtype is np.float32
                
                print(f'pts {(pts).dtype, pts.shape, pts}')
                
            else:
                #print(f'{len(rz_coords)} rz region defined')
                pts= np.array(rz_coords).astype(np.float32).reshape((-1,1,2))
                print(f'pts {(pts).dtype, pts.shape, pts}')

                
            try:
                in_rz = cv.pointPolygonTest(pts, bbox_lmidpt, False) #+1==inside, -1==outside, 0=on the edge of rz #list x,y coordinates, dtype is np.float32
                print(f'poly test result {in_rz}')
                    
            except Exception as ex:
                logging.warning(f"Error getting coordinates: {ex!r}")
                print(f"Error getting coordinates {ex!r}")


            if class_list[classid]=='person':# and (in_rz and in_rz >=0): #check if the person is inside rz
                people +=1            
                box_coords.append(box)
                class_label.append(class_list[classid])
                cv.rectangle(frame, box, color, 2) #np.array([left, top, width, height])
                cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1) #room to overlay text
                cv.putText(frame, class_list[classid], (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

                bbox_lmidpt = np.array((box[0]+(box[2]/2),(box[1]+box[3]))).astype(np.float32) #x,y of lower midpoint of each bounding box
            #cv.imwrite('./saveframes/frame_{}.jpg'.format(time.time()),frame) #should I choose to save the frames. I need to initialize i's

        print(f'num x,ys are {len(box_coords)} and person counter is {people} and {class_label}')

        ##ModBus errors out on empty or None input. e.g. at zero person detected so:
        if len(box_coords) == 0:
            box_nos = [int(0)]
        else:
            box_nos = [i for i in np.asarray(box_coords).flatten()]

        DataBank.set_words(0, [int(people)]) #write  to register 1            
        DataBank.set_words(1, box_nos) #write to register 2 
        print(f"I found: {client.read_holding_registers(0,1)} pers; bbox: x,y,h,w={client.read_holding_registers(1,40)}") #2nd param is no of inputs to read

        time.sleep(1) #'sleep' for 1 second in between.
        # if chr(cv.waitKey(1)&255) == 'q':
        #     break

        

        #cv.imshow('window', img)
        #cv.waitKey(0)
        
        cap.release()
        cv.destroyAllWindows()
        server.stop()


