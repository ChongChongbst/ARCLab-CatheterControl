import numpy as np
import cv2



def convolve(img, kernel):
    """
    img -> gray scale image stored as numpy array
    """
    ## Pad input image by mirroring intensity
    border_size_y = kernel.shape[0] // 2
    border_size_x = kernel.shape[1] // 2

    img_padded =  cv2.copyMakeBorder(img, border_size_y, border_size_y, border_size_x, border_size_x, cv2.BORDER_REFLECT)
    output = np.zeros_like(img)

    for i in range(border_size_y, img.shape[0] + border_size_y):
        for j in range(border_size_x, img.shape[1] + border_size_x):

            i_min = i - border_size_y
            i_max = i + border_size_y + 1
            j_min = j - border_size_x
            j_max = j + border_size_x + 1

            if kernel.shape[0] % 2 == 0:
                i_max -= 1
            if kernel.shape[1] % 2 == 0:
                j_max -= 1

            output[i_min, j_min] = np.sum(np.multiply(img_padded[i_min:i_max, j_min:j_max], kernel))

    return output

def computeGmGd(gx, gy):
    gm = np.sqrt(np.square(gx) + np.square(gy))
    gd = np.arctan2(gy, gx) * 180 / np.pi
    return gm, gd

def NMS(gm, gd):
    """
    Implementation of Non-maximum Suppression.

    gm -> gradient magnitude image stored as numpy array
    gd -> gradient direction image stored as numpy array
    """
    ## Round gradient direction to nearest 45 degrees
    gd = np.round(gd / 45) * 45

    output = np.copy(gm)

    ## Map gradient direction to relative pixel location in (x, y)
    gd_map = {-180: [-1, 0], 
              -135: [-1, -1],
               -90: [0, -1],
               -45: [1, -1],
                 0: [1, 0],
                45: [1, 1],
                90: [0, 1],
               135: [-1, 1],
               180: [-1, 0]}

    for y in range(gm.shape[0]):
        for x in range(gm.shape[1]):

            ## Get the relative positive gradient direction
            pgd = gd_map[gd[y, x]]

            ## Get the indices of gradient magnitudes in the positive and negative gradient directions 
            pgm_y = y + pgd[1]
            pgm_x = x + pgd[0]
            ngm_y = y - pgd[1]
            ngm_x = x - pgd[0]

            ## Limit the indices to be inside the boundaries of the image
            pgm_y = max([min([pgm_y, gm.shape[0] - 1]), 0])
            pgm_x = max([min([pgm_x, gm.shape[1] - 1]), 0])
            ngm_y = max([min([ngm_y, gm.shape[0] - 1]), 0])
            ngm_x = max([min([ngm_x, gm.shape[1] - 1]), 0])

            ## Get the gradient magnitudes in the positive and negative gradient directions 
            pgm = gm[pgm_y, pgm_x]
            ngm = gm[ngm_y, ngm_x]

            ## Suppress the gradient magnitude in the current position if it is not the largest
            if gm[y, x] < pgm or gm[y, x] < ngm:
                output[y, x] = 0

    return output



class HoughBezier:


    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = np.float32(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))


    def gaussian_smooth(self):

        ## Gaussian kernel
        k_g = np.array([[2, 4, 5, 4, 2],
                        [4, 9, 12, 9, 4],
                        [5, 12, 15, 12, 5],
                        [4, 9, 12, 9, 4],
                        [2, 4, 5, 4, 2]]) / 159

        self.image = convolve(self.image, k_g)


    def edge_detection(self):

        ## Gradient kernels
        k_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

        k_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

        ## Find x and y gradients of lane image
        image_gx = convolve(self.image, k_x)
        image_gy = convolve(self.image, k_y)

        ## Find magnitude and direction of gradient of lane image
        self.image_gm, image_gd = computeGmGd(image_gx, image_gy)

        ## Apply NMS
        #self.image_gm = NMS(self.image_gm, image_gd)


    def write_image(self, path):
        print(list(np.unique(self.image_gm)))
        cv2.imwrite(path, self.image_gm)

    
    def hough_transform(self):
        """
        Implementation of Hough Transform.
        Accumulator cells' resolution is 1 theta x 1 rho.

        eg -> edge image stored as numpy array
        """
        self.rho_max = round((self.image_gm.shape[0] ** 2 + self.image_gm.shape[1] ** 2) ** 0.5)
        rho_length = self.rho_max * 2
        self.theta_max = 90
        theta_length = 2 * self.theta_max  ## from -90 to 90, there are 181 values

        self.hough_space = np.zeros((rho_length, theta_length))

        for y in range(self.image_gm.shape[0]):
            for x in range(self.image_gm.shape[1]):

                if self.image_gm[y, x] == 0:
                    continue

                for theta in range(theta_length):
                    rho = round(x * np.sin(np.radians(theta - self.theta_max))
                              + y * np.cos(np.radians(theta - self.theta_max)))
                    self.hough_space[rho + self.rho_max, theta] += 1

    
    def bezier(p_start, p_c, p_end, s):
        return (1 - s) ** 2 * p_start + 2 * (1 - s) * s * p_c + s ** 2 * p_end


    def hough_transform_bezier(self):


        

        y_max = self.image_gm.shape[0]
        x_max = self.image_gm.shape[1]

        diag = np.sqrt(y_max ** 2 + x_max ** 2)

        #self.hough_space = np.zeros((y_max, x_max * 2 + y_max, y_max, x_max))
        self.hough_space = np.zeros((y_max, y_max, y_max - 20, x_max - 20))

        p_start_x = x_max - 1
        p_end_x = 0

        for p_start_y in range(y_max):
            for p_end_y in range(y_max):
                for p_c_y in range(10, y_max - 10):
                    for p_c_x in range(10, x_max - 10):

                        s = np.linspace(0, 1, diag)

                        p_start = np.array(p_start_x, p_start_y)
                        p_c = np.array(p_c_x, p_c_y)
                        p_end = np.array(p_end_x, p_end_y)

                        poses = (self.bezier(p_start, p_c, p_end, s)).astype(int)

                        print(poses)
















    def inverse_hough_transform(self, intensity_threshold, theta_ranges=None):
        """
        Use information in Hough Space to plot lines on original image.
        """
        y_max = self.image.shape[0]
        x_max = self.image.shape[1]
        img_with_line = np.copy(self.image)
        img_with_line = cv2.cvtColor(img_with_line, cv2.COLOR_GRAY2BGR)

        self.hough_space[self.hough_space <= intensity_threshold] = 0

        line_count = 0

        for rho in range(self.hough_space.shape[0]):
            for theta in range(self.hough_space.shape[1]):
                if self.hough_space[rho, theta] == 0:
                    continue

                r = rho - self.rho_max
                t = theta - self.theta_max

                print('rho: ', r, ' theta: ', t)

                ## Restrict theta to be in a certain range
                if theta_ranges is not None:
                    keep_line = False
                    for theta_range in theta_ranges:
                        if t >= theta_range[0] and t <= theta_range[1]:
                            keep_line = True
                            break
                    if not keep_line:
                        continue

                t = np.radians(t)
                x = r * np.sin(t)
                y = r * np.cos(t)

                x1 = x + self.rho_max * np.cos(t)
                y1 = y - self.rho_max * np.sin(t)

                x2 = x - self.rho_max * np.cos(t)
                y2 = y + self.rho_max * np.sin(t)

                ## Draw line on original image
                cv2.line(img_with_line, (round(x), round(y)), (round(x1), round(y1)), (255, 51, 153), 1)
                cv2.line(img_with_line, (round(x), round(y)), (round(x2), round(y2)), (255, 51, 153), 1)
                line_count += 1

        print('[inverseHT] Line Count: ', line_count)
        return img_with_line


    


if __name__ == '__main__':
    input_image_path = '/home/inffzy/nutstore_files/arclab_research/ARCLab-CCCatheter/data/real_experiment/images/0000_segmented.png'
    output_image_path = '/home/inffzy/Desktop/test.png'
    output_image2_path = '/home/inffzy/Desktop/test2.png'

    HB = HoughBezier(input_image_path)

    #HB.gaussian_smooth()
    HB.edge_detection()
    HB.write_image(output_image_path)
    HB.hough_transform_bezier()
    #HB.hough_transform()
    #img = HB.inverse_hough_transform(150)

    #cv2.imwrite(output_image2_path, img)

