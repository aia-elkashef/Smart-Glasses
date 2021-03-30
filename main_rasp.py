import numpy as np
import cv2
import tensorflow as tf
from yolov3.utils import detect_image, Load_Yolo_model
from PIL import Image
import os
import time

#=============== OCR  ================#
import pytesseract
def OCR(frame, lang):
    img = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filename = "/images/PreProcessed.jpg"
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.fromarray(gray), lang=lang)
    return text

#============== Text to Speech  ===========#
import speech_recognition as sr
from speech_recognation import recordAudio
from gtts import gTTS 
from playsound import playsound


#============= Face Recogntion ==================#
import face_recognition
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def Face_recognizer(frame):
    process_this_frame = True
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame == True:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            myobj = gTTS(text=name, lang='en', slow=False)
            myobj.save("OCRtext.mp3") 
            playsound("OCRtext.mp3")
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


full_path = os.path.realpath(__file__)
path = os.path.dirname(full_path)

if not os.path.exists(path + "/images"):
    os.makedirs(path + "/images")

cap = cv2.VideoCapture(0)


while True:    

    data = recordAudio()
    ret, frame = cap.read()
    #Read English Text
    if 'read' in data:
        print('Reading English...')
        cv2.imwrite(path + "/images/English.jpg", frame)
        text = OCR(frame, lang='eng').strip()
        if len(text) > 0:
           myobj = gTTS(text=text, lang='en', slow=False)
           myobj.save("OCRtext.mp3") 
           playsound("OCRtext.mp3")
        else:
            myobj = gTTS(text="No english text detected!", lang='en', slow=False)
            myobj.save("OCRtext.mp3") 
            playsound("OCRtext.mp3")
        print('Done.')
        
    #object Detection        
    elif 'object' in data:
        print('Detecting Objects...')
        image_path = path + "/images/original.jpg"
        cv2.imwrite(image_path, frame)
        # class_ids, close_objs = detect_objects(frame)
        yolo = Load_Yolo_model()
        image, objects = detect_image(yolo, image_path, path + "/images/output.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
        objects = set(objects)
        objects = ' and '.join(objects)
        myobj = gTTS(text=objects, lang='en', slow=False)
        myobj.save("objects.mp3") 
        playsound("objects.mp3")

    #Face Recogntion
    elif 'face' in data:
        print("Face Recogntion")
        Face_recognizer(frame)
        
    # cv2.imshow("object detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'): 
        break 

cap.release()          
cv2.destroyAllWindows()
