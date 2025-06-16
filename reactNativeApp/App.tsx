// App.tsx
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  FlatList,
  Alert,
  SafeAreaView,
  StatusBar,
  ActivityIndicator,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface Exercise {
  name: string;
  selected: boolean;
}

const DEFAULT_SERVER_URL = 'ws://192.168.1.100:8765';
const STORAGE_KEY_SERVER_URL = 'server_url';

const App: React.FC = () => {
  const [serverUrl, setServerUrl] = useState<string>('');
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [connecting, setConnecting] = useState<boolean>(false);
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [selectedExercise, setSelectedExercise] = useState<string>('');
  const [reps, setReps] = useState<string>('10');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>('');

  // Zuletzt verwendete Server-URL laden
  useEffect(() => {
    const loadServerUrl = async () => {
      try {
        const savedUrl = await AsyncStorage.getItem(STORAGE_KEY_SERVER_URL);
        if (savedUrl) {
          setServerUrl(savedUrl);
        } else {
          setServerUrl(DEFAULT_SERVER_URL);
        }
      } catch (error) {
        console.error('Fehler beim Laden der Server-URL:', error);
        setServerUrl(DEFAULT_SERVER_URL);
      }
    };

    loadServerUrl();
  }, []);

  // WebSocket-Verbindung aufbauen
  const connectToServer = () => {
    if (socket) {
      socket.close();
    }

    try {
      setConnecting(true);
      setStatus('Verbindung wird hergestellt...');
      
      const ws = new WebSocket(serverUrl);
      
      ws.onopen = () => {
        console.log('WebSocket-Verbindung geöffnet');
        setConnected(true);
        setConnecting(false);
        setStatus('Verbunden');
        
        // Server-URL speichern
        AsyncStorage.setItem(STORAGE_KEY_SERVER_URL, serverUrl)
          .catch(error => console.error('Fehler beim Speichern der URL:', error));
        
        // Übungen abrufen
        ws.send(JSON.stringify({ type: 'list_exercises' }));
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'exercises') {
            const exerciseList = message.data.map((name: string) => ({
              name,
              selected: false,
            }));
            setExercises(exerciseList);
            setStatus(`${exerciseList.length} Übungen verfügbar`);
          } 
          else if (message.type === 'status') {
            if (message.status === 'started') {
              setIsRunning(true);
              setStatus(`Übung läuft: ${message.exercise} (${message.reps} Wdh.)`);
            } else if (message.status === 'stopped') {
              setIsRunning(false);
              setStatus('Übung beendet');
            }
          }
          else if (message.type === 'error') {
            Alert.alert('Fehler', message.message);
            setStatus('Fehler: ' + message.message);
          }
        } catch (error) {
          console.error('Fehler beim Verarbeiten der Nachricht:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket-Fehler:', error);
        setConnecting(false);
        setStatus('Verbindungsfehler');
        Alert.alert(
          'Verbindungsfehler',
          'Verbindung zum Server fehlgeschlagen. Bitte überprüfe die Server-Adresse und versuche es erneut.'
        );
      };
      
      ws.onclose = () => {
        console.log('WebSocket-Verbindung geschlossen');
        setConnected(false);
        setStatus('Verbindung getrennt');
      };
      
      setSocket(ws);
    } catch (error) {
      console.error('Fehler beim Verbinden:', error);
      setConnecting(false);
      setStatus('Verbindungsfehler');
      Alert.alert('Fehler', 'Fehler beim Herstellen der Verbindung');
    }
  };

  // Übung starten
  const startExercise = () => {
    if (!socket || !connected) {
      Alert.alert('Fehler', 'Keine Verbindung zum Server');
      return;
    }
    
    if (!selectedExercise) {
      Alert.alert('Fehler', 'Bitte wähle eine Übung aus');
      return;
    }
    
    const repetitions = parseInt(reps, 10);
    if (isNaN(repetitions) || repetitions <= 0) {
      Alert.alert('Fehler', 'Bitte gib eine gültige Anzahl an Wiederholungen ein');
      return;
    }
    
    socket.send(JSON.stringify({
      type: 'start',
      exercise: selectedExercise,
      reps: repetitions
    }));
  };

  // Übung stoppen
  const stopExercise = () => {
    if (socket && connected) {
      socket.send(JSON.stringify({ type: 'stop' }));
    }
  };

  // Übung auswählen
  const selectExercise = (exercise: string) => {
    setSelectedExercise(exercise);
    setExercises(exercises.map(ex => ({
      ...ex,
      selected: ex.name === exercise
    })));
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <View style={styles.header}>
        <Text style={styles.title}>Fitness Trainer</Text>
        <Text style={styles.subtitle}>{status}</Text>
      </View>
      
      <View style={styles.serverSection}>
        <TextInput
          style={styles.serverInput}
          value={serverUrl}
          onChangeText={setServerUrl}
          placeholder="Server-URL (ws://...)"
          placeholderTextColor="#888"
        />
        <TouchableOpacity 
          style={[styles.button, styles.connectButton]} 
          onPress={connectToServer}
          disabled={connecting}
        >
          {connecting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Verbinden</Text>
          )}
        </TouchableOpacity>
      </View>
      
      {connected && (
        <>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Verfügbare Übungen</Text>
            <FlatList
              data={exercises}
              keyExtractor={item => item.name}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.exerciseItem,
                    item.selected ? styles.exerciseItemSelected : null
                  ]}
                  onPress={() => selectExercise(item.name)}
                >
                  <Text style={[
                    styles.exerciseText,
                    item.selected ? styles.exerciseTextSelected : null
                  ]}>
                    {item.name}
                  </Text>
                </TouchableOpacity>
              )}
              style={styles.exerciseList}
            />
          </View>
          
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Wiederholungen</Text>
            <TextInput
              style={styles.repsInput}
              value={reps}
              onChangeText={setReps}
              keyboardType="number-pad"
              placeholder="Anzahl der Wiederholungen"
              placeholderTextColor="#888"
            />
          </View>
          
          <View style={styles.controlSection}>
            {isRunning ? (
              <TouchableOpacity 
                style={[styles.button, styles.stopButton]} 
                onPress={stopExercise}
              >
                <Text style={styles.buttonText}>Übung beenden</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity 
                style={[styles.button, styles.startButton]} 
                onPress={startExercise}
                disabled={!selectedExercise}
              >
                <Text style={styles.buttonText}>Übung starten</Text>
              </TouchableOpacity>
            )}
          </View>
        </>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D',
    padding: 15,
  },
  header: {
    marginBottom: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f72585',
    marginBottom: 5,
    textShadowColor: '#f72585',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#ccc',
  },
  serverSection: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  serverInput: {
    flex: 1,
    height: 50,
    borderRadius: 8,
    paddingHorizontal: 15,
    backgroundColor: '#222',
    color: '#fff',
    marginRight: 10,
  },
  connectButton: {
    backgroundColor: '#4361ee',
    height: 50,
    paddingHorizontal: 15,
    justifyContent: 'center',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 10,
  },
  exerciseList: {
    maxHeight: 200,
  },
  exerciseItem: {
    paddingVertical: 12,
    paddingHorizontal: 15,
    borderRadius: 8,
    backgroundColor: '#222',
    marginBottom: 8,
  },
  exerciseItemSelected: {
    backgroundColor: '#f72585',
  },
  exerciseText: {
    color: '#fff',
    fontSize: 16,
  },
  exerciseTextSelected: {
    fontWeight: 'bold',
  },
  repsInput: {
    height: 50,
    borderRadius: 8,
    paddingHorizontal: 15,
    backgroundColor: '#222',
    color: '#fff',
  },
  controlSection: {
    marginTop: 'auto',
    marginBottom: 20,
  },
  button: {
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  startButton: {
    backgroundColor: '#f72585',
    shadowColor: '#f72585',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 8,
    elevation: 6,
  },
  stopButton: {
    backgroundColor: '#e63946',
    shadowColor: '#e63946',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 8,
    elevation: 6,
  },
});

export default App;