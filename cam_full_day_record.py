# import the opencv library 
import cv2 
import time
import os
import numpy as np
from datetime import datetime, timedelta

cam_ports = ['/dev/video6', '/dev/video0']
width = 640
height = 480
save_dir = './data/snapshots_day/'
DELAY = 15 * 60  # 15 minutes
DELAY_SHORT = 60  # 1 minute
WARM_UP = 10

# Set the start and end times for capturing snapshots
start_time = datetime.now()
end_time = start_time + timedelta(days=1)

while True:
    try:
        # define a video capture object
        vids = [cv2.VideoCapture(p) for p in cam_ports]

        for _ in range(WARM_UP):
            frames = []
            for vid in vids:
                frame = vid.read()[1]
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # frame = vid.read()[1]
                frames.append(frame)
            image = cv2.hconcat(frames)

            # Display the resulting frame 
            cv2.imshow('snapshot_frame', image) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # # Wait for 15 seconds
        # time.sleep(15)

        # Save the captured frame as an image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, frame in enumerate(frames):
            image_name = os.path.join(save_dir, f"snapshot_{timestamp}_{i}.jpg")
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(image_name, frame)
            print(f"Snapshot saved: {image_name}")


        # # After the loop release the cap object 
        for vid in vids:
            vid.release()
        # Wait
        time.sleep(DELAY)
    except Exception as e:
        print(f"Cound't acquire images at {datetime.now().strftime('%Y%m%d_%H%M%S')}: {e}")
        # Wait less
        time.sleep(DELAY_SHORT)


# Destroy all the windows 
cv2.destroyWindow('snapshot_frame')

