import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("../dlibExamples/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarkd = dlib_facelandmark(gray,face)
        print(face_landmarkd)
        for n in range(0,68):
            x = face_landmarkd.part(n).x
            y = face_landmarkd.part(n).y
            cv2.circle(frame,(x,y),1,(0,255,255),1)
    cv2.imshow("Face Landmarks", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()