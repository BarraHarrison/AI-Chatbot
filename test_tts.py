import pyttsx3

engine = pyttsx3.init(driverName='nsss')
engine.say("Testing text to speech")
engine.runAndWait()
