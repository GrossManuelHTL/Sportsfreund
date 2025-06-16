# websocket_server.py
import asyncio
import json
import logging
import websockets
import subprocess
import os
import sys

logging.basicConfig(level=logging.INFO)

# Anpassungsfähiger Pfad für verschiedene Betriebssysteme
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTIVE_TRAINER_PATH = os.path.join(BASE_DIR, "interactive_trainer.py")


class TrainerServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.active_process = None
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        logging.info(f"Client verbunden. Aktive Verbindungen: {len(self.clients)}")

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logging.info(f"Client getrennt. Aktive Verbindungen: {len(self.clients)}")

    async def send_to_all(self, message):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients])

    async def list_exercises(self):
        """Verfügbare Übungen auflisten"""
        exercises_dir = os.path.join(BASE_DIR, 'exercises')
        available_exercises = []

        try:
            for item in os.listdir(exercises_dir):
                item_path = os.path.join(exercises_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                    available_exercises.append(item)
        except Exception as e:
            logging.error(f"Fehler beim Auflisten der Übungen: {str(e)}")

        return available_exercises

    async def start_exercise(self, exercise_name, reps):
        """Übung mit dem interaktiven Trainer starten"""
        if self.active_process and self.active_process.poll() is None:
            self.active_process.terminate()
            await asyncio.sleep(1)

        logging.info(f"Starte Übung: {exercise_name} mit {reps} Wiederholungen")

        cmd = ["python", "main.py", "--exercise", exercise_name, "--reps", str(reps), "--interactive"]

        try:
            self.active_process = subprocess.Popen(cmd)

            await self.send_to_all(json.dumps({
                "type": "status",
                "status": "started",
                "exercise": exercise_name,
                "reps": reps
            }))

            

            return True
        except Exception as e:
            logging.error(f"Fehler beim Starten der Übung: {str(e)}")
            await self.send_to_all(json.dumps({
                "type": "error",
                "message": f"Fehler beim Starten der Übung: {str(e)}"
            }))
            return False

    async def stop_exercise(self):
        """Laufende Übung beenden"""
        if self.active_process and self.active_process.poll() is None:
            self.active_process.terminate()
            await self.send_to_all(json.dumps({
                "type": "status",
                "status": "stopped"
            }))

    async def handle_client(self, websocket):
        # Beachte: path-Parameter entfernt
        await self.register(websocket)
        try:
            # Sende verfügbare Übungen beim Verbinden
            exercises = await self.list_exercises()
            await websocket.send(json.dumps({
                "type": "exercises",
                "data": exercises
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data["type"] == "start":
                        exercise_name = data.get("exercise")
                        reps = data.get("reps", 10)
                        await self.start_exercise(exercise_name, reps)

                    elif data["type"] == "stop":
                        await self.stop_exercise()

                    elif data["type"] == "list_exercises":
                        exercises = await self.list_exercises()
                        await websocket.send(json.dumps({
                            "type": "exercises",
                            "data": exercises
                        }))

                except json.JSONDecodeError:
                    logging.error(f"Ungültiges JSON-Format: {message}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def start_server(self):
        # Ohne 'path' Parameter verwenden
        async with websockets.serve(
                lambda websocket: self.handle_client(websocket),
                self.host,
                self.port
        ):
            logging.info(f"Server läuft auf {self.host}:{self.port}")
            await asyncio.Future()  # Läuft für immer


def main():
    server = TrainerServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logging.info("Server durch Benutzer beendet.")


if __name__ == "__main__":
    main()