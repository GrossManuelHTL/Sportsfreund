import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, Button, SafeAreaView } from 'react-native';
import exerciseInstructions from './exerciseInstructions.json';

interface ExerciseInstructionProps {
  route: {
    params: {
      exerciseType: 'squats' | 'push_ups';
    }
  };
  navigation: any;
}

const ExerciseInstruction: React.FC<ExerciseInstructionProps> = ({ route, navigation }) => {
  const { exerciseType } = route.params;
  const [reps, setReps] = useState('10');

  const instructions = exerciseInstructions[exerciseType];

  if (!instructions) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.errorText}>Instructions not found</Text>
        <Button title="Go Back" onPress={() => navigation.goBack()} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>
        {exerciseType === 'squats' ? 'Squat' : 'Push-up'} Instructions
      </Text>

      <View style={styles.repsContainer}>
        <Text style={styles.repsLabel}>How many repetitions?</Text>
        <TextInput
          style={styles.repsInput}
          value={reps}
          onChangeText={setReps}
          keyboardType="numeric"
          maxLength={3}
        />
      </View>

      <ScrollView style={styles.instructionsContainer}>
        <Text style={styles.sectionTitle}>Preparation</Text>
        {instructions.preparation_instructions.map((instruction, index) => (
          <Text key={`prep-${index}`} style={styles.instruction}>
            • {instruction}
          </Text>
        ))}

        <Text style={styles.sectionTitle}>Execution</Text>
        {instructions.execution_instructions.map((instruction, index) => (
          <Text key={`exec-${index}`} style={styles.instruction}>
            • {instruction}
          </Text>
        ))}

        <Text style={styles.repsSummary}>
          Perform {reps} repetition{parseInt(reps) !== 1 ? 's' : ''}.
        </Text>
      </ScrollView>

      <View style={styles.buttonContainer}>
        <Button title="Start Exercise" onPress={() => {
          // Here you would add logic to start the exercise tracking
          navigation.navigate('ExerciseTracking', {
            exerciseType,
            reps: parseInt(reps)
          });
        }} />

        <Button title="Cancel" onPress={() => navigation.goBack()} />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  repsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    justifyContent: 'center',
  },
  repsLabel: {
    fontSize: 18,
    marginRight: 8,
  },
  repsInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    padding: 8,
    width: 60,
    fontSize: 18,
    textAlign: 'center',
  },
  instructionsContainer: {
    flex: 1,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 8,
  },
  instruction: {
    fontSize: 16,
    marginBottom: 8,
    lineHeight: 22,
  },
  repsSummary: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 16,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 18,
    color: 'red',
    marginBottom: 16,
    textAlign: 'center',
  },
  buttonContainer: {
    marginVertical: 10,
  }
});

export default ExerciseInstruction;