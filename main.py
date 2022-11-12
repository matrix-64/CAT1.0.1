import cv2
import mcat1

cap = cv2.VideoCapture(0)

global cat

while cap.isOpened():
    success, image = cap.read()
    if success:
        cat = mcat1.CAT(image.shape)
        break

while cap.isOpened():
    success, video = cap.read()
    if not success :
        continue
    cat.operate(video)
    if cv2.waitKey(1) == 27:
        cat.stop()
        break

cap.release()
