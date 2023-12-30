# Handles Image File Making

import os
from datetime import datetime
import numpy as np

class Image():
    def __init__(self, width, height, color_data):
        """ takes an image width, image height, a list of RGB vectors, and a file type and generates an image object """
        self.width = width
        self.height = height
        self.color_data = np.array(color_data)
        
    def save(self, file_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f"), file_type = 'ppm'):
        """ saves the image object's data to an image file """
        # "Building a Ray Tracer in Python" Series by Arun Ravindran "ArunRocks" on Youtube
        
        # make folder for images if images folder does not exist
        if not os.path.exists("Images"):
            os.mkdir("Images")
        
        file_path = "Images\{0}.{1}".format(file_name, file_type)
        
        # rename images that have the same name to file_name(i) to prevent overwriting files
        i = 0
        while os.path.exists(file_path):
            i += 1
            file_path = "Images\{0}({1}).{2}".format(file_name, i, file_type)
        
        # resize the array into a 2D array of color vectors for indexing
        resized_color_data = self.color_data.reshape([self.width, self.height, 3])
        
        # create new image files and write data
        # note: use PIL image library to convert ppm to png later
        with open(file_path, "w+") as file:
            if (file_type == 'ppm'):
                # ppm file setup
                file.write("P3\n" + "{0} {1}\n".format(self.width, self.height) + "255\n")
                
                # write data
                # note: look into using numpy functions to save to a file later
                for x in range(self.width):
                    for y in range(self.height):
                        for c in range(3):
                            file.write(str(resized_color_data[x][y][c]))
                            file.write(" ")
                        file.write("  ")
                    file.write("\n")
            else:
                raise Exception("Cannot save image. '{0}' is not a supported file type.".format(file_type))
        print("Saved Image Successfully")