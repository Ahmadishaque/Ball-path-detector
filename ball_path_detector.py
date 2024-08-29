import cv2
import numpy as np
from kalman_filter import KalmanFilter

videoframe = cv2.VideoCapture(0)
kf = KalmanFilter()

prevBall = None
distance = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

while True:
    ret, frame = videoframe.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (17, 17), 0)

    balls = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=75, maxRadius=400)

    if balls is not None:
        balls = np.uint16(np.around(balls))
        
        chosen = None

        for i in balls[0, :]:
            if chosen is None:
                chosen = i

            if prevBall is not None:
                if distance(chosen[0], chosen[1], prevBall[0], prevBall[1]) >= distance(i[0], i[1], prevBall[0], prevBall[1]):
                    chosen = i

        if chosen is not None:
            detected_x, detected_y = chosen[0], chosen[1]
            
            kf.update(np.array([[detected_x], [detected_y]]))

            predicted_x, predicted_y = kf.predict()

            cv2.circle(frame, (detected_x, detected_y), chosen[2], (255, 0, 255), 3)

            cv2.circle(frame, (int(predicted_x), int(predicted_y)), chosen[2], (0, 255, 0), 3)

            prevBall = chosen

    cv2.imshow("Ball on screen", frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

videoframe.release()
cv2.destroyAllWindows()
