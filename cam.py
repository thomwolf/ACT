# import the opencv library 
import cv2 

cam_ports = ['/dev/video6', '/dev/video0']
width = 640
height = 480

# define a video capture object 
vids = [cv2.VideoCapture(p) for p in cam_ports]

while(True): 
    
    # Capture the video frame 
    # by frame 
    frames = [vid.read()[1] for vid in vids]
    frame = cv2.hconcat(frames)
    # Display the resulting frame 
    cv2.imshow('rame', frame) 
    
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object 
for vid in vids:
    vid.release()
# Destroy all the windows 
cv2.destroyAllWindows() 
