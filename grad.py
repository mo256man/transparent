import cv2
import numpy as np
import math

height, width = 300, 260

grad = np.zeros((height, width), np.uint8)

for h in range(height):
	v = int(255/2 * (math.sin(2 * math.pi * h / height)+1))
	grad[h:h+1, :] = v

while True:
	grad = np.roll(grad, 1, axis=0)
	cv2.imshow("grad", grad)
	if cv2.waitKey(1) == 27:
		break
cv2.destroyAllWindows()
