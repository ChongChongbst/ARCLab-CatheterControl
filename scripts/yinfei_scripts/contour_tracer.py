import cv2
import numpy as np



class ContourTracer:

    def __init__(self, image_path):
        """
        Args:
            image_path (path string to png file): reading path to image for which a contour will be drawn
        """
        self.image_path = image_path


    def trace_contour(self, threshold=200, contour_index=-1, resize=(640, 480)):
        """
        Output the points corresponding to the contour on a given image using OpenCV
        
        Args:
            threshold (float): threshold value for cv2.threshold
            contour_index (int): the choice of contour when multiple are present
            resize (tuple of 2 ints): size of resized input image 
        
        Returns:
            ((n, 2) numpy array): n 2D points that make up the contour
        """
        self.img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        self.img = self.img[:, :, :3]

        ## Resize image
        self.resized_img = cv2.resize(self.img, resize)

        #convert img to grey
        img_grey = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)

        #get threshold image
        _, thresh_img = cv2.threshold(img_grey, threshold, 255, cv2.THRESH_BINARY)

        #find contours
        self.contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_np = np.squeeze(self.contours[contour_index], axis=1)

        return contours_np


    def draw_contour(self, output_image_path):
        """
        Save the contour as an image

        Args:
            output_image_path (path string to png file): writing path to image showing the contour
        """
        #create an empty image for contours
        img_contours = np.zeros(self.resized_img.shape)
        
        #draw the contours on the empty image
        cv2.drawContours(img_contours, self.contours, -1, (0, 255, 0), 3)
        
        #save image
        cv2.imwrite(output_image_path, img_contours) 

    
    def draw_resized_image(self, resized_image_path):
        """
        Save the resized input image

        Args:
            resized_image_path (path string to png file): writing path to resized image
        """
        cv2.imwrite(resized_image_path, self.resized_img) 
