Ones the program is initiated each image is sequentially loaded
and the model is trained.

the Program then accesses the camera and recognises the face in the frame.
originally my face was at 30cm from the camera and the program correctly identified it.
Ones the distance between the face and the camera increased to more than 50cm, the program
gave a false output.

conclusion: the code is good enough to recognise the faces which are closer to the camera (30cm and less) with high accuracy.
but as the distance increases the accuracy decreases drastically.