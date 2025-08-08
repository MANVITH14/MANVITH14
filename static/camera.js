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
    fpsCounter.textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastTime = now;
  }
}

async function checkCameraStatus() {
  try {
    const response = await fetch('/camera_status');
    const data = await response.json();
    return data.camera_available;
  } catch (error) {
    console.error('Failed to check camera status:', error);
    return false;
  }
}

startBtn.addEventListener('click', async () => {
  hideErrors();
  
  // Check if camera is available
  const cameraAvailable = await checkCameraStatus();
  if (!cameraAvailable) {
    noCameraDiv.classList.remove('hidden');
    return;
  }
  
  // Start camera feed
  cameraActive = true;
  startBtn.disabled = true;
  startBtn.classList.add('disabled');
  stopBtn.disabled = false;
  stopBtn.classList.remove('disabled');
  
  cameraContainer.classList.remove('hidden');
  
  // Reset FPS counter
  frameCount = 0;
  lastTime = Date.now();
  fpsInterval = setInterval(updateFPS, 100);
  
  // Add timestamp to video feed URL to force refresh
  videoFeed.src = `/video_feed?t=${Date.now()}`;
  
  videoFeed.onload = () => {
    if (cameraActive) {
      // Refresh the image continuously for real-time feed
      setTimeout(() => {
        if (cameraActive) {
          videoFeed.src = `/video_feed?t=${Date.now()}`;
        }
      }, 33); // ~30 FPS
    }
  };
  
  videoFeed.onerror = () => {
    if (cameraActive) {
      showError('Failed to load camera feed. Please try again.');
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