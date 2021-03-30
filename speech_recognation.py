#!/usr/bin/env python

import speech_recognition as sr
import os
import os

def recordAudio():
    r = sr.Recognizer()           # Record Audio


    # Speech recognition using Google Speech Recognition
    data = ""
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            # Uses the default API key
            # To use another API key: `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            data = r.recognize_google(audio)
            print("You said: " + data)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return data

if __name__ == '__main__':
    print('listening...')
    data = recordAudio()
    print(data)







