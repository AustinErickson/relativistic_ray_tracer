# https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

import joblib
from skimage.io import imread
from skimage.transform import resize
import os

def resize_all(src, pklname, include, width, height):
    """ load an image, resize the image, create an array that stores image information and store in a dictionary. Writes dictionary to pickle file."""
    
    # data initialization
    data = dict()
    data["description"] = "resized images in RGB format"
    data["label"] = []
    data["filename"] = []
    data["data"] = []
    
    pklname = f"{pklname}_{width}x{height}px.pkl"
    
    # read images, resize and write to target directory
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
            
            for file in os.listdir(current_path):
                if file[-3:] in {"jpg", "png"}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))
                    data["label"].append(subdir[:4])
                    data["filename"].append(file)
                    data["data"].append(im)
                    
        joblib.dump(data, pklname)

HOMEPATH = os.path.expanduser('~')
data_path = "{0}/desktop/data_set/AnimalFace/Image".format(HOMEPATH)
print(os.listdir(data_path))
