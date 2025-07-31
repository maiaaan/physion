import os, pathlib
import numpy as np
import sys
sys.path.append("./src")
from skimage import io
from physion.visual_stim.main import visual_stim, init_bg_image


########################################
#######  ----    ASAHI    ----  ########
########################################

params = {"Image-ID":0}

def get_images_as_array():
    
    IM_FOLDER = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'Asahi_bank')
    
    AI_img = []

    if os.path.isdir(IM_FOLDER):
        for filename in np.sort(os.listdir(IM_FOLDER)):
            image_path = os.path.join(IM_FOLDER, filename)
            img = io.imread(image_path, as_gray=True)
            AI_img.append(img)
        return AI_img
    else:
        print(' [!!]  Asahi Images folder not found !!! [!!]  ')
        return [np.ones((10,10))*0.5 for i in range(5)]

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        # initializing set of NI
        self.AI_img = get_images_as_array()

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        return np.rot90(\
                self.AI_img[int(self.experiment['Image-ID'][index])-1], 
                        k=1)

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('asahi')
    print(params)

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
