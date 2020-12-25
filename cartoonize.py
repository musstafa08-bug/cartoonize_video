import cv2
import numpy as np

cap =  cv2.VideoCapture(0)

while(True):
    _, img = cap.read()
    x = img.reshape((-1, 5))
    x = np.float32(x)

    criteria = cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.95
    k = 10

    # center allocation
    _, label, center = cv2.kmeans(x, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)


    center = np.uint8(center)
    flat = center[label.flatten()]
    resolution = flat.reshape((img.shape))

    cv2.imshow("Cartoonized", resolution)

    if cv2.waitKey(3) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()