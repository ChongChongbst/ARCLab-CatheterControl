import numpy as np
import cv2
import torch
import sys
import path_settings
sys.path.insert(0, path_settings.scripts_dir) 

from reconstruction_scripts.reconst_sim_opt2pts import reconstructCurve



class PostProcessing:

    def __init__(self, input_image_path):
        self.input_image_path = input_image_path

    def run(self, segmented_image_path):

        img = cv2.imread(self.input_image_path)
        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        cv2.imshow("Raw Img", img)
        frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (0, 50, 100), (98, 255, 255))
        kernel = np.ones((6, 6), np.uint8)
        img = cv2.erode(frame_threshold, kernel, iterations=5)
        img = cv2.dilate(img, kernel, iterations=5)
        img = cv2.erode(frame_threshold, kernel, iterations=5)


        
        #cv2.imshow("Segmentation", img)
        #cv2.waitKey(0)

        img = cv2.resize(img, (640, 480))
        cv2.imwrite(segmented_image_path, img)

        '''
        # im_bin = cv2.cvtColor(dilatation_dst, cv2.CV_32F)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        skel = np.zeros(img.shape, np.uint8)

        # Repeat steps 2-4
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img)==0:
                break

        # skel = np.zeros(img.shape, np.uint8)
        # skel = cv2.bitwise_or(skel, dilatation_dst)
        #skeleton = cv2.cvtColor(skel, cv2.CV_8U)
        print("fine")
        cv2.imshow("Skeleton", skel)
        cv2.waitKey(0)
        #cv2.imwrite(path+str(i)+'.jpeg', dilatation_dst)
        '''

    def test_reconstruction(self, image_path):

        ### ground truth bezier points : P0
        l = 0.2
        p_0 = torch.tensor([0.02, 0.002, 0.0])
        bezier_specs_torch = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.19168896], dtype=torch.float)
        bezier_specs_init_torch = torch.tensor([0.01957763, 0.00191553, 0.09690971, -0.03142124, -0.00828425, 0.18168159],
            dtype=torch.float, requires_grad=True)
        loss_weight = torch.tensor([1.0, 1.0, 1.0])

        ## Detect actual bezier
        bezier_reconstruction = reconstructCurve(image_path, l, p_0, bezier_specs_torch, bezier_specs_init_torch, loss_weight, total_itr=200, img_width=640, img_height=480)
        bezier_reconstruction.getOptimize(None, p_0)
        bezier_reconstruction.plotProjCenterline()

        ## Convert actual bezier to cc
        optimized_bezier_specs = bezier_reconstruction.para.detach().numpy().reshape((2, 3))
        print(optimized_bezier_specs)



if __name__ == '__main__':
    input_image_path = '/home/arclab/Desktop/catheter_control0.jpeg'
    segmented_image_path = '/home/arclab/Desktop/recon_input.png'
    post_processing = PostProcessing(input_image_path)
    post_processing.run(segmented_image_path)
    post_processing.test_reconstruction(segmented_image_path)

