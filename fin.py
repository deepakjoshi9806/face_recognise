import face_recognition
from sklearn import svm
import os
import cv2
import numpy as np

encodings = []
names = []
present = []


# function for face recognition and training


def face_recognize(dir) :
    if dir[-1] != '/' :
        dir += '/'
    train_dir = os.listdir(dir)
    for person_img in train_dir :
        face = face_recognition.load_image_file(
            dir + person_img)  # load an image from the database
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1 :
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)  # generate encodings for the loaded face and save it in encodings array
            names.append(person_img)  # save names in names array
        else :
            print(person_img + " can't be used for training")
    print("training complete\n ")
    clf = svm.SVC(gamma='scale')  # run the svm classifier
    clf.fit(encodings, names)  # label encodings with the respective names
    video_capture = cv2.VideoCapture(0)  # start video stream
    while True :
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, : :-1]  # convert bgr to rgb
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations :  # extract location of face in a frame of video stream
            test_image_enc = face_recognition.face_encodings(rgb_frame[top :bottom, left :right])  # generate encoding
            name = clf.predict(test_image_enc)  # compare this encoding with the above generated encoding
            print(*name)  # print the label of closest matching encoding
            present.append(*name)  # save the label of closest matching encoding
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break


def main() :
    train_dir = "C:/resources"  # load directory where image database is stored
    print("process initiated! \n")
    face_recognize(train_dir)


# save names of all students in a txt document


a_file = open("test.txt", "w")
for names in present :
    np.savetxt(a_file, names)
a_file.close()

if __name__ == "__main__" :
    main()
