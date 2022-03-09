
import os
#os.chdir(os.getcwd())
from image_detect import detectCount
import configparser
import numpy as np


#######################Set seed#######################
def set_seed(seed = None):
    '''Sets the seed for REPRODUCIBILITY.'''
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(0)



def main():
    ''' Executes the livestream_detect file'''
    #print(help(detectCount))
    #os.chdir(os.getcwd())
    config = configparser.ConfigParser()
    config.read('./persondetectapp/src/model_data/config/config_live_cam.config') #reads in the config file that has inference params below

    #Arguments values for livestream_detect.py inference file.
    config_path=config.get("MODELINFO", "config_path")
    
    weights_path=config.get("MODELINFO", "weights_path")
    class_path=config.get("MODELINFO", "class_path")    
    width=config.getfloat("MODELINFO", "width")
    height=config.getfloat("MODELINFO", "height")
    conf_thres=config.getfloat("MODELINFO", "conf_thres")
    #nms_thres=config.getfloat("MODELINFO", "nms_thres")
    host=config.get("MODBUSINFO", "host")
    port=config.getint("MODBUSINFO", "port")
    #print('from main', conf_thres)


    people = detectCount(class_path, weights_path, width, height, host, port)
    #print('I am getting Started')
    people.stream_count()

if __name__ == '__main__':
    main()



