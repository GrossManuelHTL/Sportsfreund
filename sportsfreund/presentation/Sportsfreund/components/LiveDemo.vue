<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const steps = [
  { icon: '🏃‍♂️', text: 'Person performs a squat' },
  { icon: '🤖', text: 'Sportsfreund detects the movement' },
  { icon: '💡', text: 'AI analyzes the posture' },
  { icon: '🔊', text: 'Feedback is spoken: "Great job! Go a bit deeper."' }
]

const currentStep = ref(0)
const isAnimating = ref(false)
const animationType = ref('')

const nextStep = () => {
  if (isAnimating.value) return
  
  isAnimating.value = true
  animationType.value = 'slide-left'
  
  setTimeout(() => {
    currentStep.value = (currentStep.value + 1) % steps.length
    setTimeout(() => {
      isAnimating.value = false
      animationType.value = ''
    }, 100)
  }, 300)
}

const prevStep = () => {
  if (isAnimating.value) return
  
  isAnimating.value = true
  animationType.value = 'slide-right'
  
  setTimeout(() => {
    currentStep.value = currentStep.value === 0 ? steps.length - 1 : currentStep.value - 1
    setTimeout(() => {
      isAnimating.value = false
      animationType.value = ''
    }, 100)
  }, 300)
}

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'w' || event.key === 'W') {
    event.preventDefault()
    nextStep()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div class="keyboard-demo">
    <div class="step-container">
      <div 
        class="step-display"
        :class="{ 
          'animating': isAnimating,
          'slide-left': animationType === 'slide-left',
          'slide-right': animationType === 'slide-right'
        }"
      >
        <div class="step-icon">{{ steps[currentStep].icon }}</div>
        <div class="step-text">{{ steps[currentStep].text }}</div>
      </div>
      
      <div class="step-indicator">
        <div 
          v-for="(step, index) in steps"
          :key="index"
          class="indicator-dot"
          :class="{ 'active': index === currentStep }"
        ></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.keyboard-demo {
  text-align: center;
  margin: 2rem 0;
}

.step-container {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(16, 185, 129, 0.1));
  border: 3px solid rgba(37, 99, 235, 0.3);
  border-radius: 20px;
  padding: 4rem 3rem;
  min-height: 250px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
}

.step-display {
  transition: all 0.7s cubic-bezier(0.25, 0.8, 0.25, 1);
  transform: translateX(0) scale(1) rotateY(0deg);
  opacity: 1;
}

/* Noch coolere 3D-Slide-Animationen */
.step-display.animating.slide-left {
  transform: translateX(-150px) scale(0.7) rotateY(-20deg);
  opacity: 0;
  filter: blur(3px);
}

.step-display.animating.slide-right {
  transform: translateX(150px) scale(0.7) rotateY(20deg);
  opacity: 0;
  filter: blur(3px);
}

.step-icon {
  font-size: 5rem;
  margin-bottom: 2rem;
  display: block;
  transition: all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
  transform: scale(1) rotate(0deg);
  text-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
}

.step-display.animating .step-icon {
  transform: scale(1.5) rotate(360deg);
  filter: drop-shadow(0 0 20px rgba(37, 99, 235, 0.8));
}

.step-text {
  font-size: 1.8rem;
  font-weight: 700;
  color: #2563eb;
  line-height: 1.4;
  transition: all 0.7s ease;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  letter-spacing: 0.5px;
}

.step-display.animating .step-text {
  letter-spacing: 4px;
  color: #2563eb;
  text-shadow: 0 0 10px rgba(37, 99, 235, 0.5);
}

.step-indicator {
  display: flex;
  justify-content: center;
  gap: 0.8rem;
  margin-top: 2rem;
}

.indicator-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #cbd5e1;
  transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  position: relative;
}

.indicator-dot.active {
  background: linear-gradient(45deg, #2563eb, #16a34a);
  transform: scale(1.6);
  box-shadow: 
    0 0 30px rgba(37, 99, 235, 0.8),
    0 0 60px rgba(37, 99, 235, 0.4),
    inset 0 0 10px rgba(255, 255, 255, 0.3);
}

.indicator-dot.active::before {
  content: '';
  position: absolute;
  top: -5px;
  left: -5px;
  right: -5px;
  bottom: -5px;
  border: 3px solid rgba(37, 99, 235, 0.4);
  border-radius: 50%;
  animation: pulse-ring 2s infinite;
}

.indicator-dot.active::after {
  content: '';
  position: absolute;
  top: -8px;
  left: -8px;
  right: -8px;
  bottom: -8px;
  border: 1px solid rgba(37, 99, 235, 0.2);
  border-radius: 50%;
  animation: pulse-ring 2s infinite 0.5s;
}

@keyframes pulse-ring {
  0% {
    transform: scale(0.8);
    opacity: 1;
  }
  100% {
    transform: scale(2);
    opacity: 0;
  }
}

.navigation-hint {
  margin-top: 1.5rem;
  font-size: 0.95rem;
  color: #6b7280;
  font-weight: 500;
  padding: 0.5rem 1rem;
  background: rgba(107, 114, 128, 0.1);
  border-radius: 20px;
  display: inline-block;
  transition: all 0.3s ease;
}

.navigation-hint:hover {
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  transform: translateY(-1px);
}

/* Zusätzliche Highlight-Animation für Demo-Box */
.step-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(37, 99, 235, 0.1),
    transparent
  );
  transition: left 0.8s ease;
}

.step-display.animating ~ .step-container::before {
  left: 100%;
}

@media (max-width: 768px) {
  .step-container {
    padding: 2rem 1rem 2rem 1rem;
    min-height: 150px;
  }
  
  .step-icon {
    font-size: 3rem;
  }
  
  .step-text {
    font-size: 1.2rem;
  }
  
  .navigation-hint {
    font-size: 0.85rem;
  }
}
</style>
