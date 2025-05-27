import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator, StackScreenProps } from '@react-navigation/stack';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView } from 'react-native';
import ExerciseInstruction from './exerciseInstruction';

// Define the param list for type safety
type RootStackParamList = {
  Home: undefined;
  ExerciseInstruction: { exerciseType: 'squats' | 'push_ups' };
  ExerciseTracking: { exerciseType: string; reps: number };
};

// Type the screen props
type ExerciseTrackingProps = StackScreenProps<RootStackParamList, 'ExerciseTracking'>;
type HomeScreenProps = StackScreenProps<RootStackParamList, 'Home'>;

// Fixed component with proper typing
const ExerciseTracking: React.FC<ExerciseTrackingProps> = ({ route }) => {
  const { exerciseType, reps } = route.params;
  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Exercise Tracking</Text>
      <Text style={styles.text}>
        Tracking {exerciseType}: {reps} repetitions
      </Text>
    </SafeAreaView>
  );
};

// Home screen component with proper typing
const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>SportsFreund</Text>
      <Text style={styles.subtitle}>Choose an exercise to begin</Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ExerciseInstruction', { exerciseType: 'squats' })}
        >
          <Text style={styles.buttonText}>Squats</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ExerciseInstruction', { exerciseType: 'push_ups' })}
        >
          <Text style={styles.buttonText}>Push-ups</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

// Typed stack navigator
const Stack = createStackNavigator<RootStackParamList>();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="ExerciseInstruction"
          component={ExerciseInstruction as React.ComponentType<any>}
          options={{ title: "Exercise Instructions" }}
        />
        <Stack.Screen
          name="ExerciseTracking"
          component={ExerciseTracking}
          options={{ title: "Exercise Tracking" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    marginBottom: 32,
    textAlign: 'center',
  },
  buttonContainer: {
    width: '100%',
    alignItems: 'center',
  },
  button: {
    backgroundColor: '#4A90E2',
    width: '80%',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  text: {
    fontSize: 18,
    marginBottom: 16,
  }
});

export default App;