import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const SquatsScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>This is a squat</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
  }
});

export default SquatsScreen;