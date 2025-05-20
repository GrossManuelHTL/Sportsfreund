import MFRC522
import signal
import json
import os

continue_reading = True
last_uid = None
json_file = "uids.json"

# UID zu String konvertieren
def uidToString(uid):
    return ''.join(format(i, '02X') for i in uid)

# Beenden mit Ctrl+C
def end_read(signal, frame):
    global continue_reading
    print("\nCtrl+C captured, ending read.")
    continue_reading = False

# Signalhandler setzen
signal.signal(signal.SIGINT, end_read)

# RFID-Leser initialisieren
MIFAREReader = MFRC522.MFRC522()

# Hauptloop
print("RFID-Schreiber gestartet – Lege eine neue Karte auf.")
print("Drücke Ctrl-C zum Beenden.")

while continue_reading:
    (status, TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

    if status == MIFAREReader.MI_OK:
        (status, uid) = MIFAREReader.MFRC522_SelectTagSN()

        if status == MIFAREReader.MI_OK:
            current_uid = uidToString(uid)

            if current_uid != last_uid:
                last_uid = current_uid
                print(f"\nNeue UID erkannt: {current_uid}")
                description = input("Beschreibung eingeben: ")

                # Falls Datei existiert, öffnen und Inhalt laden
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        try:
                            uid_map = json.load(f)
                        except json.JSONDecodeError:
                            uid_map = {}
                else:
                    uid_map = {}

                # Neue UID einfügen
                uid_map[current_uid] = description

                # Datei überschreiben
                with open(json_file, "w") as f:
                    json.dump(uid_map, f, indent=4)

                print("Gespeichert.")
