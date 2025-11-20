import rerun as rr
import numpy as np

rr.init("test_app")
rr.spawn()

# Log some 3D points
points = np.random.randn(100, 3)
rr.log("points", rr.Points3D(points))