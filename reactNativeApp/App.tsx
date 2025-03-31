import React, { useState } from 'react';
import { View, Button, Text, StyleSheet } from 'react-native';

export default function App() {
    const [isRunning, setIsRunning] = useState(false);

    const handleStart = () => {
        // Here you could make an API call to your backend
        // to start a process, or just set state.
        setIsRunning(true);
    };

    const handleStop = () => {
        // Similarly, you could stop something on the backend
        setIsRunning(false);
    };

    return (
        <View style={styles.container}>
            <Text>{isRunning ? 'Running...' : 'Stopped'}</Text>
            <Button title="Start" onPress={handleStart} />
            <Button title="Stop" onPress={handleStop} />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
});