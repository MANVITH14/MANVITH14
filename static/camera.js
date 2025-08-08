const startBtn = document.getElementById('start-camera');
const stopBtn = document.getElementById('stop-camera');
const cameraContainer = document.getElementById('camera-container');
const videoFeed = document.getElementById('video-feed');
const errorDiv = document.getElementById('error');
const noCameraDiv = document.getElementById('no-camera');
const fpsCounter = document.getElementById('fps-counter');

let cameraActive = false;
let fpsInterval;
let frameCount = 0;
let lastTime = Date.now();
let demoMode = false;

function showError(message) {
  errorDiv.textContent = message;
  errorDiv.classList.remove('hidden');
  noCameraDiv.classList.add('hidden');
}

function hideErrors() {
  errorDiv.classList.add('hidden');
  noCameraDiv.classList.add('hidden');
}

function updateFPS() {
  frameCount++;
  const now = Date.now();
  const elapsed = now - lastTime;
  
  if (elapsed >= 1000) {
    const fps = Math.round((frameCount * 1000) / elapsed);
    const mode = demoMode ? 'DEMO' : '';
    fpsCounter.textContent = `FPS: ${fps} ${mode}`;
    frameCount = 0;
    lastTime = now;
  }
}

async function checkCameraStatus() {
  try {
    const response = await fetch('/camera_status');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to check camera status:', error);
    return { camera_available: false, demo_mode: true, message: 'Connection error' };
  }
}

function updateStartButtonText(status) {
  if (status.demo_mode) {
    startBtn.textContent = 'Start Demo Mode';
    startBtn.title = 'No camera available - will show demo with sample images';
  } else {
    startBtn.textContent = 'Start Camera';
    startBtn.title = 'Start live camera feed';
  }
}

startBtn.addEventListener('click', async () => {
  hideErrors();
  
  // Check camera status
  const status = await checkCameraStatus();
  demoMode = status.demo_mode;
  
  if (!status.camera_available && !status.demo_mode) {
    noCameraDiv.classList.remove('hidden');
    return;
  }
  
  // Start camera/demo feed
  cameraActive = true;
  startBtn.disabled = true;
  startBtn.classList.add('disabled');
  stopBtn.disabled = false;
  stopBtn.classList.remove('disabled');
  
  cameraContainer.classList.remove('hidden');
  
  // Update camera info based on mode
  const cameraInfoP = document.querySelector('.camera-info p');
  if (demoMode) {
    cameraInfoP.textContent = 'Demo mode: Cycling through sample images with age/gender detection.';
  } else {
    cameraInfoP.textContent = 'Live detection running. Age and gender predictions are shown in real-time.';
  }
  
  // Reset FPS counter
  frameCount = 0;
  lastTime = Date.now();
  fpsInterval = setInterval(updateFPS, 100);
  
  // Add timestamp to video feed URL to force refresh
  videoFeed.src = `/video_feed?t=${Date.now()}`;
  
  videoFeed.onload = () => {
    if (cameraActive) {
      // Refresh the image continuously for real-time feed
      const refreshRate = demoMode ? 100 : 33; // Slower for demo mode
      setTimeout(() => {
        if (cameraActive) {
          videoFeed.src = `/video_feed?t=${Date.now()}`;
        }
      }, refreshRate);
    }
  };
  
  videoFeed.onerror = () => {
    if (cameraActive) {
      showError('Failed to load video feed. Please try again.');
      stopCamera();
    }
  };
});

stopBtn.addEventListener('click', () => {
  stopCamera();
});

function stopCamera() {
  cameraActive = false;
  startBtn.disabled = false;
  startBtn.classList.remove('disabled');
  stopBtn.disabled = true;
  stopBtn.classList.add('disabled');
  
  cameraContainer.classList.add('hidden');
  
  if (fpsInterval) {
    clearInterval(fpsInterval);
    fpsInterval = null;
  }
  
  fpsCounter.textContent = 'FPS: --';
  videoFeed.src = '';
  
  hideErrors();
}

// Initialize camera status check on page load
document.addEventListener('DOMContentLoaded', async () => {
  const status = await checkCameraStatus();
  updateStartButtonText(status);
  
  if (status.demo_mode) {
    // Show a helpful message about demo mode
    const subtitle = document.querySelector('.subtitle');
    subtitle.textContent = 'Demo mode: No camera detected. Will use sample images to demonstrate detection.';
    subtitle.style.color = '#fbbf24'; // Amber color for demo mode
  }
});

// Handle page visibility change to stop camera when tab is hidden
document.addEventListener('visibilitychange', () => {
  if (document.hidden && cameraActive) {
    stopCamera();
  }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
  if (cameraActive) {
    stopCamera();
  }
});