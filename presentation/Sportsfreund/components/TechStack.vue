<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const technologies = [
  { 
    icon: 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg', 
    name: 'Python', 
    description: 'Main Language for Processing & Backend',
    isPrimary: false
  },
  { 
    icon: 'https://opencv.org/wp-content/uploads/2022/05/logo.png', 
    name: 'OpenCV', 
    description: 'Computer Vision & Key Point Detection',
    isPrimary: false
  },
  { 
    icon: 'https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg', 
    name: 'Scikit-Learn', 
    description: 'Machine Learning Algorithms',
    isPrimary: false
  },
  { 
    icon: 'https://www.tensorflow.org/images/tf_logo_social.png', 
    name: 'TensorFlow', 
    description: 'Deep Learning & Pose Estimation',
    isPrimary: false
  },
  { 
    icon: 'https://nodejs.org/static/images/logo.svg', 
    name: 'Node.js', 
    description: 'Backend & Database Management',
    isPrimary: false
  }
]

const currentTech = ref(0)
const isAnimating = ref(false)
const animationType = ref('')

const nextTech = () => {
  if (isAnimating.value) return
  
  isAnimating.value = true
  animationType.value = 'zoom-in'
  
  setTimeout(() => {
    currentTech.value = (currentTech.value + 1) % technologies.length
    setTimeout(() => {
      isAnimating.value = false
      animationType.value = ''
    }, 100)
  }, 400)
}

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'q' || event.key === 'Q') {
    event.preventDefault()
    nextTech()
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
  <div class="tech-demo">
    <div class="tech-container">
      <div 
        class="tech-display"
        :class="{ 
          'animating': isAnimating,
          'slide-animation': animationType === 'slide-next'
        }"
      >
        <div class="tech-icon-container">
          <img 
            :src="technologies[currentTech].icon" 
            :alt="technologies[currentTech].name"
            class="tech-icon"
          />
        </div>
        <h3 class="tech-name">{{ technologies[currentTech].name }}</h3>
        <p class="tech-description">{{ technologies[currentTech].description }}</p>
      </div>
      
      <div class="tech-indicator">
        <div 
          v-for="(tech, index) in technologies"
          :key="index"
          class="indicator-dot"
          :class="{ 'active': index === currentTech }"
        ></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tech-demo {
  text-align: center;
  margin: 2rem 0;
}

.tech-container {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(16, 185, 129, 0.1));
  border: 3px solid rgba(37, 99, 235, 0.3);
  border-radius: 20px;
  padding: 4rem 3rem;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
}

.tech-display {
  transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
  transform: translateX(0) scale(1) rotateY(0deg);
  opacity: 1;
}

.tech-display.animating.slide-animation {
  transform: translateX(-100px) scale(0.8) rotateY(-15deg);
  opacity: 0;
  filter: blur(2px);
}

.tech-icon-container {
  height: 100px;
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.tech-display.animating .tech-icon-container {
  transform: scale(1.3) rotate(180deg);
}

.tech-icon {
  max-height: 100px;
  max-width: 200px;
  object-fit: contain;
  transition: all 0.6s ease;
  filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
}

.tech-display.animating .tech-icon {
  filter: drop-shadow(0 0 20px rgba(37, 99, 235, 0.8));
}

.tech-name {
  font-size: 2.5rem;
  font-weight: 700;
  color: #2563eb;
  margin-bottom: 1rem;
  transition: all 0.6s ease;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.tech-display.animating .tech-name {
  letter-spacing: 3px;
  color: #16a34a;
  text-shadow: 0 0 10px rgba(37, 99, 235, 0.5);
}

.tech-description {
  font-size: 1.3rem;
  font-weight: 500;
  color: #64748b;
  line-height: 1.4;
  margin-bottom: 1rem;
  transition: all 0.6s ease;
}

.primary-badge {
  background: linear-gradient(45deg, #2563eb, #16a34a);
  color: white;
  padding: 0.5rem 1.5rem;
  border-radius: 25px;
  font-size: 0.9rem;
  font-weight: 600;
  display: inline-block;
  margin-top: 1rem;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

.tech-indicator {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2.5rem;
}

.indicator-dot {
  width: 14px;
  height: 14px;
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
    0 0 60px rgba(37, 99, 235, 0.4);
}

.indicator-dot.active::before {
  content: '';
  position: absolute;
  top: -6px;
  left: -6px;
  right: -6px;
  bottom: -6px;
  border: 2px solid rgba(37, 99, 235, 0.4);
  border-radius: 50%;
  animation: pulse-ring 2s infinite;
}

@keyframes pulse-ring {
  0% {
    transform: scale(0.8);
    opacity: 1;
  }
  100% {
    transform: scale(2.2);
    opacity: 0;
  }
}

/* Sweep-Effekt für Container */
.tech-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(37, 99, 235, 0.15),
    transparent
  );
  transition: left 0.8s ease;
}

.tech-display.animating ~ .tech-container::before {
  left: 100%;
}

@media (max-width: 768px) {
  .tech-container {
    padding: 3rem 2rem;
    min-height: 250px;
  }
  
  .tech-icon-container {
    height: 80px;
  }
  
  .tech-icon {
    max-height: 80px;
    max-width: 160px;
  }
  
  .tech-name {
    font-size: 2rem;
  }
  
  .tech-description {
    font-size: 1.1rem;
  }
  
  .indicator-dot {
    width: 12px;
    height: 12px;
  }
}
</style>
