# base path to YOLO directory
MODEL_PATH = "yolo"
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3
# Threading ON/OFF. Please refer 'mylib>thread.py'.
Thread = True
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
url = 0
# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = True
# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 400
MIN_DISTANCE = 200
