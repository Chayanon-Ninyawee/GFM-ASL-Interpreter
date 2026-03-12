const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");

const letterDiv = document.getElementById("letter");
const wordDiv = document.getElementById("word");

function resizeCanvas() {
  const rect = canvasElement.getBoundingClientRect();

  canvasElement.width = rect.width;
  canvasElement.height = rect.height;
}

window.addEventListener("resize", resizeCanvas);

resizeCanvas();

let model = null;

async function loadModel() {
  try {
    model = await tf.loadGraphModel("web_model/model.json");
    console.log("ML model loaded");
  } catch (e) {
    console.log("No ML model found, using heuristic");
    console.log(e);
  }
}

loadModel();

// smoothing buffer

let history = [];

function smoothLetter(letter) {
  history.push(letter);

  if (history.length > 10) history.shift();

  const freq = {};

  for (const l of history) freq[l] = (freq[l] || 0) + 1;

  let best = "-";
  let max = 0;

  for (const k in freq) {
    if (freq[k] > max) {
      max = freq[k];
      best = k;
    }
  }

  return best;
}

// word builder

let currentWord = "";
let lastLetter = "-";
let stableFrames = 0;

function updateWord(letter) {
  if (letter === lastLetter) stableFrames++;
  else stableFrames = 0;

  if (stableFrames === 8 && letter !== "-") {
    currentWord += letter;
    stableFrames = 0;
  }

  lastLetter = letter;

  return currentWord;
}

function clearWord() {
  currentWord = "";
  wordDiv.innerText = "";
}

// fallback detection if no ML model

function heuristicLetter(landmarks) {
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  const indexUp = indexTip.y < landmarks[6].y;
  const middleUp = middleTip.y < landmarks[10].y;
  const ringUp = ringTip.y < landmarks[14].y;
  const pinkyUp = pinkyTip.y < landmarks[18].y;

  if (!indexUp && !middleUp && !ringUp && !pinkyUp) return "A";

  if (indexUp && middleUp && ringUp && pinkyUp) return "B";

  if (indexUp && !middleUp && !ringUp && !pinkyUp) return "L";

  return "-";
}

function predictLetter(landmarks) {
  if (!model) return heuristicLetter(landmarks);

  return tf.tidy(() => {
    const input = [];

    // const wrist = landmarks[0];

    // for (const p of landmarks) {
    //   input.push(p.x - wrist.x);
    //   input.push(p.y - wrist.y);
    //   input.push(p.z - wrist.z);
    // }

    for (const p of landmarks) {
      input.push(p.x);
      input.push(p.y);
      input.push(p.z);
    }

    const tensor = tf.tensor(input).reshape([1, 63, 1]);

    const prediction = model.execute(tensor);

    const index = prediction.argMax(1).dataSync()[0];

    const labels = [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "J",
      "K",
      "L",
      "M",
      "N",
      "O",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z",
      "del",
      "nothing",
      "space",
    ];

    console.log(input.length);

    return labels[index];
  });
}

// mediapipe setup

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 0,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },

  width: 640,
  height: 480,
});

camera.start();

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

    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
      color: "#00ffaa",
      lineWidth: 1.5,
    });

    drawLandmarks(canvasCtx, landmarks, {
      color: "#ff5555",
      radius: 2,
    });

    let letter = predictLetter(landmarks);

    letter = smoothLetter(letter);

    const word = updateWord(letter);

    letterDiv.innerText = letter;
    wordDiv.innerText = word;
  }

  canvasCtx.restore();
}
