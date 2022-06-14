
# import the opencv library
import os
import cv2
  

img_save_dir = '/home/inffzy/Desktop/ARCLab/ARCLab-CCCatheter/data/real_experiments/cam_test_images' 


# define a video capture object
vid = cv2.VideoCapture(0)


i = 0
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)

    if i % 30 == 0:
        img_file = str(i // 30).zfill(4) + '.png' 

    cv2.imwrite(os.path.join(img_save_dir, img_file), frame)


    i += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()