const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const outputDiv = document.getElementById("output");

// Initialize MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  },
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});

hands.onResults(onResults);

// Start camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 480,
  height: 360,
});
camera.start();

// ===============================
// Simple Letter Detection Logic
// ===============================

function detectLetter(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  const indexUp = indexTip.y < landmarks[6].y;
  const middleUp = middleTip.y < landmarks[10].y;
  const ringUp = ringTip.y < landmarks[14].y;
  const pinkyUp = pinkyTip.y < landmarks[18].y;

  // A: all fingers down
  if (!indexUp && !middleUp && !ringUp && !pinkyUp) {
    return "A";
  }

  // B: all fingers up
  if (indexUp && middleUp && ringUp && pinkyUp) {
    return "B";
  }

  // L: index up, others down
  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    return "L";
  }

  // C: thumb close to index (curve shape)
  const distance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

  if (distance < 0.05) {
    return "C";
  }

  return "-";
}

// ===============================

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height,
  );

  if (results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    const letter = detectLetter(landmarks);
    outputDiv.innerText = "Detected Letter: " + letter;
  } else {
    outputDiv.innerText = "Detected Letter: -";
  }

  canvasCtx.restore();
}
