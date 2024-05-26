const socket = io('/');
const peer = new Peer();
let myVideoStream;
let myId;
var videoGrid = document.getElementById('videoDiv')
var myvideo = document.createElement('video');
myvideo.muted = true;
const peerConnections = {}
let frameBuffer = [];
const MAX_FRAMES = 25;  // Number of frames to capture in each interval

navigator.mediaDevices.getUserMedia({
  video: true,
  audio: true
}).then((stream) => {
  myVideoStream = stream;
  addVideo(myvideo, stream);
  peer.on('call', call => {
    call.answer(stream);
    const vid = document.createElement('video');
    call.on('stream', userStream => {
      addVideo(vid, userStream);
    })
    call.on('error', (err) => {
      alert(err)
    })
    call.on("close", () => {
      console.log(vid);
      vid.remove();
    })
    peerConnections[call.peer] = call;
  })

  // Start capturing frames
  const captureInterval = setInterval(() => {
    captureFrames(myvideo);
  }, 80);  // Capture frames every 80 milliseconds (12.5 fps)

  const sendInterval = setInterval(() => {
    if (frameBuffer.length >= MAX_FRAMES) {
      sendFrames();
    }
  }, 2000);  // Process frames every 2 seconds

}).catch(err => {
  alert(err.message)
})

peer.on('open', (id) => {
  myId = id;
  socket.emit("newUser", id, roomID);
})
peer.on('error', (err) => {
  alert(err.type);
});
socket.on('userJoined', id => {
  console.log("new user joined")
  const call = peer.call(id, myVideoStream);
  const vid = document.createElement('video');
  call.on('error', (err) => {
    alert(err);
  })
  call.on('stream', userStream => {
    addVideo(vid, userStream);
  })
  call.on('close', () => {
    vid.remove();
    console.log("user disconnect")
  })
  peerConnections[id] = call;
})
socket.on('userDisconnect', id => {
  if (peerConnections[id]) {
    peerConnections[id].close();
  }
})

function addVideo(video, stream) {
  video.srcObject = stream;
  video.addEventListener('loadedmetadata', () => {
    video.play()
  })
  videoGrid.append(video);
}

function captureFrames(video) {
  if (frameBuffer.length < MAX_FRAMES) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      frameBuffer.push(blob);

      if (frameBuffer.length >= MAX_FRAMES) {
        sendFrames();
        frameBuffer = [];  // Clear the buffer after sending
      }
    }, 'image/jpeg');
  }
}

function sendFrames() {
  const formData = new FormData();
  frameBuffer.forEach((frame, index) => {
    formData.append('frames[]', frame, `frame${index}.jpg`);
  });

  fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      console.log('Predicted class:', data.class);
      displayPrediction(data.class);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

function displayPrediction(predictedClass) {
  const classesNames = ["right", "left", "rotate up", "rotate down", "downright", "right-down", "clockwise", "counter clock", "zeta", "cross"];
  const predictionElement = document.getElementById('prediction');
  if (!predictionElement) {
    const newPredictionElement = document.createElement('div');
    newPredictionElement.id = 'prediction';
    newPredictionElement.innerText = `Predicted Hand Movement: ${predictedClass ? classesNames[predictedClass] : predictedClass}`;
    document.body.appendChild(newPredictionElement);
  } else {
    predictionElement.innerText = `Predicted Hand Movements: ${predictedClass ? classesNames[predictedClass] : predictedClass}`;
  }
}
