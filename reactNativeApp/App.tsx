import React, { useState, useRef } from 'react';
import {
  Animated,
  StyleSheet,
  Text,
  TouchableWithoutFeedback,
  View,
} from 'react-native';

const App: React.FC = () => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const scaleValue = useRef(new Animated.Value(1)).current;

  const handleToggle = () => {
    // Subtle "pop" animation when the user taps the button
    Animated.sequence([
      Animated.timing(scaleValue, {
        toValue: 1.1,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleValue, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setIsRunning(!isRunning);
    });
  };

  // A simple triangle for the "play" symbol
  const PlayIcon = () => (
      <View style={styles.playIconContainer}>
        <View style={styles.playTriangle} />
      </View>
  );

  // Two vertical bars for the "pause" symbol
  const PauseIcon = () => (
      <View style={styles.pauseIconContainer}>
        <View style={styles.pauseBar} />
        <View style={styles.pauseBar} />
      </View>
  );

  return (
      <View style={styles.container}>
        <TouchableWithoutFeedback onPress={handleToggle}>
          <Animated.View
              style={[styles.bigButton, { transform: [{ scale: scaleValue }] }]}
          >
            {isRunning ? <PauseIcon /> : <PlayIcon />}
          </Animated.View>
        </TouchableWithoutFeedback>

        <Text style={styles.statusText}>
          {isRunning ? 'Running' : 'Stopped'}
        </Text>
      </View>
  );
};

export default App;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D', // Dark background for high contrast
    alignItems: 'center',
    justifyContent: 'center',
  },
  bigButton: {
    width: 250,
    height: 250,
    borderRadius: 125,
    backgroundColor: '#f72585', // Vibrant neon-pink color
    borderWidth: 4,
    borderColor: '#fff',

    // Center icon within the circle
    alignItems: 'center',
    justifyContent: 'center',

    // Neon-like shadow/glow
    shadowColor: '#f72585',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
    elevation: 15,
  },
  statusText: {
    marginTop: 30,
    fontSize: 32,
    fontWeight: '700',
    color: '#fff',

    // Subtle neon glow for the text
    textShadowColor: '#f72585',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },

  // --- Play Icon ---
  playIconContainer: {
    width: 80,
    height: 80,
    // The container itself is centered in the bigButton, so no absolute positioning needed
    alignItems: 'center',
    justifyContent: 'center',
  },
  playTriangle: {
    width: 0,
    height: 0,
    borderStyle: 'solid',

    // A 40×60 triangle: 40 wide, 60 tall
    borderLeftWidth: 40,
    borderTopWidth: 30,
    borderBottomWidth: 30,
    borderLeftColor: '#fff',
    borderTopColor: 'transparent',
    borderBottomColor: 'transparent',
  },

  // --- Pause Icon ---
  pauseIconContainer: {
    width: 80,
    height: 80,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pauseBar: {
    width: 10,
    height: 50,
    backgroundColor: '#fff',
    marginHorizontal: 5,
  },
});
