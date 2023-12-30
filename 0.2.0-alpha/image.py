# Handles Image File Making

import os
import numpy as np

class Image():
    def __init__(self, image_width, image_height, name = "ray-trace", file_type = "ppm"):
        self.width = image_width
        self.height = image_height
        self.data = np.full([image_width, image_height, 3], np.array([0, 0, 0]))
        
        self.name = name
        self.file_type = file_type
        
        print("Created Image Successfully")
        
    def save(self):
        # "Building a Ray Tracer in Python" Series by Arun Ravindran "ArunRocks" on Youtube
        """ save the image to a file """
        file_name = "{0}.{1}".format(self.name, self.file_type)
        
        # delete old image files
        if os.path.exists(file_name):
            os.remove(file_name)
        
        # create new image files and write data
        with open(file_name, "w+") as file:
            # set up file
            file.write("P3\n" + "{0} {1}\n".format(self.width, self.height) + "255\n")
            
            # write data
            for y in range(self.height):
                for x in range(self.width):
                    for c in range(3):
                        file.write(str(self.data[x][y][c]))
                        file.write(" ")
                    file.write("  ")
                file.write("\n")
        print("Saved Image Successfully")