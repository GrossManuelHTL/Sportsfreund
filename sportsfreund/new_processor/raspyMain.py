import MFRC522
import signal
import json
import requests
import time
import RPi.GPIO as GPIO
import subprocess


API_BASE = "http://localhost:3000/sportsfreund"  # anpassen falls nötig
BUTTON_PIN = 17  # Oder der GPIO-Pin, den dein Taster tatsächlich nutzt


continue_reading = True
session_id = None
current_exercise_id = None

def raspymain():
    global session_id
    session_id = startSession()
    continue_reading = True
    last_uid = None
    exercise = None

    signal.signal(signal.SIGINT, end_read)
    MIFAREReader = MFRC522.MFRC522()

    with open("uids.json", "r") as f:
        uid_map = json.load(f)

    print("RFID reader started. Press Ctrl-C to stop.")

    while continue_reading:
        (status, TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

        if status == MIFAREReader.MI_OK:
            (status, uid) = MIFAREReader.MFRC522_SelectTagSN()

            if status == MIFAREReader.MI_OK:
                current_uid = uidToString(uid)

                if current_uid != last_uid:
                    last_uid = current_uid
                    description = uid_map.get(current_uid, "Unbekannte Karte")
                    if description == "exit":
                        shutdown_pressed()
                    print("Beschreibung:", description)

                    if exercise != description:
                        exercise = description
                        startExercise(exercise)

    GPIO.cleanup()

def startSession():
    print("Starte neue Session...")
    try:
        response = requests.post(f"{API_BASE}/session", json={"start": time.strftime("%Y-%m-%dT%H:%M:%S")})
        response.raise_for_status()
        session = response.json()
        print(f"Session gestartet mit ID: {session}")
        return session["sessionID"]
    except requests.RequestException as e:
        print("Fehler beim Starten der Session:", e)
        return None


def startExercise(exercise_name):
    global session_exercise_id
    print(f"Starte Übung: {exercise_name}")
    if not session_id or exercise_name == "Unbekannte Karte":
        return

    try:
        response = requests.post(
            f"http://localhost:3000/sportsfreund/sessions/exercise",
            json={
                "sessionID": session_id,
                "exerciseID": 1,
                "feedback": "",
                "repetitions": 0,
                "score": 0,
                "exersuce": "squats",
                "session": session_id
            }
        )
        response.raise_for_status()
        result = response.json()

        print(result)
        print("jetzt hamme das gestartet")
        cmd = ['python3', 'main.py', '--exercise', exercise_name, '--interactive']
        process = subprocess.Popen(cmd)
        return process

    except requests.RequestException as e:
        print("Fehler beim Starten der Übung:", e)





def uidToString(uid):
    return ''.join(format(i, '02X') for i in uid)


def end_read(signal, frame):
    global continue_reading
    print("Ctrl+C captured, beende Lesevorgang.")
    continue_reading = False


def shutdown_pressed():
    global continue_reading
    print("Taster gedrückt: Beende aktuelle Übung, Session und Backend...")

    continue_reading = False  # Hauptschleife beenden

    time.sleep(1)

    try:
        # Sende Shutdown-Request ans Backend
        response = requests.post(f"http://localhost:3000/sportsfreund/shutdown")
        print("Backend wurde informiert:", response.text)
    except requests.RequestException as e:
        print("Fehler beim Beenden des Backends:", e)

    print("Beende Frontend...")
    exit(0)  # Beendet dieses Skript


if __name__ == "__main__":
    raspymain()
