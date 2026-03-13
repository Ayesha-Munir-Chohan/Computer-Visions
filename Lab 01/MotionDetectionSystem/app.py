import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)
first_frame = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_diff = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_diff,25,255,cv2.THRESH_BINARY)[1]
    contours,_ = cv2.findContours(thresh.copy(),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("motion_log.txt","a") as f:
            f.write("Motion detected at "+now+"\n")

    cv2.imshow("Motion Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()