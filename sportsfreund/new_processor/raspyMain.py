import MFRC522
import signal
import json
import requests
import time

API_BASE = "http://localhost:3000/sportsfreund"  # anpassen falls nötig

continue_reading = True
session_id = None
current_exercise_id = None
session_exercise_id = None


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
                    print("Beschreibung:", description)

                    if exercise != description:
                        endExercise(exercise)
                        exercise = description
                        startExercise(exercise)


def startSession():
    print("Starte neue Session...")
    try:
        response = requests.post(f"{API_BASE}/session", json={"start": time.strftime("%Y-%m-%dT%H:%M:%S")})
        response.raise_for_status()
        session = response.json()
        print(f"Session gestartet mit ID: {session['id']}")
        return session["id"]
    except requests.RequestException as e:
        print("Fehler beim Starten der Session:", e)
        return None


def endSession():
    print("Beende Session...")
    try:
        response = requests.patch(f"{API_BASE}/session/{session_id}/endtime", json={})
        response.raise_for_status()
        print("Session beendet.")
    except requests.RequestException as e:
        print("Fehler beim Beenden der Session:", e)


def startExercise(exercise_name):
    global session_exercise_id
    print(f"Starte Übung: {exercise_name}")
    if not session_id or exercise_name == "Unbekannte Karte":
        return

    try:
        response = requests.post(
            f"{API_BASE}/sessions/exercise",
            json={
                "session": session_id,
                "exercise": exercise_name,
                "start": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
        )
        response.raise_for_status()
        result = response.json()
        session_exercise_id = result["id"]
    except requests.RequestException as e:
        print("Fehler beim Starten der Übung:", e)


def endExercise(exercise_name):
    global session_exercise_id
    if not session_exercise_id:
        return
    print(f"Beende Übung: {exercise_name}")
    try:
        response = requests.patch(
            f"{API_BASE}/sessions/exercise/{session_exercise_id}/endtime",  # Diese Route müsstest du ggf. noch erstellen!
            json={"end": time.strftime("%Y-%m-%dT%H:%M:%S")}
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print("Fehler beim Beenden der Übung:", e)
    session_exercise_id = None


def uidToString(uid):
    return ''.join(format(i, '02X') for i in uid)


def end_read(signal, frame):
    global continue_reading
    print("Ctrl+C captured, beende Lesevorgang.")
    continue_reading = False
    endExercise("aktuelle Übung")
    endSession()


if __name__ == "__main__":
    raspymain()
