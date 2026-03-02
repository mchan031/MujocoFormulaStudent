import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import cv2

model_path = "models/one_car.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    # last_print_time = 0
    
    while True:
        mujoco.mj_step(m, d)
