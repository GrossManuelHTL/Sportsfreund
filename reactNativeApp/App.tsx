import React, { useState } from 'react';
import { View, Button, Text, StyleSheet, TouchableOpacity } from 'react-native';

export default function App() {
  const [isRunning, setIsRunning] = useState(false);

  const handleToggle = () => {
    setIsRunning(!isRunning);
  };

  return (
      <View style={styles.container}>
        <Text style={styles.statusText}>{isRunning ? 'Running...' : 'Stopped'}</Text>
        <TouchableOpacity style={styles.bigButton} onPress={handleToggle}>
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start'}</Text>
        </TouchableOpacity>
      </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f0f0',
  },
  statusText: {
    fontSize: 24,
    marginBottom: 20,
  },
  bigButton: {
    width: 200,
    height: 200,
    backgroundColor: '#007BFF',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 100,
  },
  buttonText: {
    color: '#fff',
    fontSize: 24,
  },
});