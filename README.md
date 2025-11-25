# 🖐️ Hand Movement Recognition
**Interactive Hand Gesture Recognition with CNN + Optical Flow and Real-Time Video Call Integration**

This project implements a hybrid system consisting of:
- A **Python Flask server** performing hand gesture recognition using a dual-input CNN (RGB + Optical Flow).
- A **Node.js web application** that manages a video-call interface and sends frames to the Python backend for real-time classification.

The model was trained **from scratch** using the *Northwestern Hand Gesture Dataset*, iteratively improved until reaching the final checkpoint:  
`northwestern_classifier53.pt`.

---

## 🚀 Overview

The goal of this project was to create an **interactive video-call interface** capable of triggering automatic actions based on **hand gestures** detected from a webcam feed.

The system integrates:

- **OpenCV** → Frame capture + optical flow generation  
- **PyTorch** → Dual-stream CNN trained on RGB & Flow  
- **Flask API** → `/predict` endpoint for gesture classification  
- **Node.js + WebSockets + PeerJS** → Real-time video-call pipeline  
- **Client-side JS** → Sends batches of frames every ~25 frames to the Flask server

---

## 🧠 Model Overview

The neural network is a **two-stream CNN**:

- **Stream 1:** RGB key frames  
- **Stream 2:** Optical flow frames (OpenCV)  
- **Fusion:** Outputs concatenated → fully connected layers → gesture prediction

Model trained using `Training_the_network.ipynb`.

Final checkpoint:  
`northwestern_classifier53.pt`.

---

## 📁 Repository Structure

```bash
HandMovementRecognition/
│── HandRecognitionPythonServer/
│   ├── Training_the_network.ipynb
│   ├── cnnModelHands.py
│   ├── flaskServer.py
│   ├── frameProcessing.py
│   ├── northwestern_classifier53.pt
│
│── NodeWebapp/
│   ├── public/
│   │   └── videocall.js
│   ├── routes/
│   │   └── videoRouter.js
│   ├── views/
│   └── server.js
│
└── .gitignore
```

## 🔧 System Architecture
1. Client (Browser)

Captures webcam frames → stores them in a buffer → sends them to Flask.
```bash
function sendFrames() {
  const formData = new FormData();
  frameBuffer.forEach((frame, index) => {
    formData.append('frames[]', frame, `frame${index}.jpg`);
  });

  fetch('https://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      console.log('Predicted class:', data.class);
      displayPrediction(data.class);
    })
    .catch(error => console.error('Error:', error));
}
```

2. Node.js Server (Video Call App)
```bash
const express = require('express');
const app = express();

const server = require('http').Server(app);
const io = require('socket.io')(server);
const port = process.env.PORT || 3000;

const { ExpressPeerServer } = require('peer');
const peer = ExpressPeerServer(server, { debug: true });

const videoRouter = require('./routes/videoRouter');

app.use('/peerjs', peer);
app.set('views', 'views');
app.set('view engine', 'ejs');
app.use(express.static('public'));

app.get('/', (req, res) => {
  res.redirect(`/meeting/home`);
});

// Video routes
app.use('/meeting', videoRouter);

io.on('connection', (socket) => {
  socket.on('newUser', (id, room) => {
    socket.join(room);
    socket.broadcast.to(room).emit("userJoined", id);

    socket.on('disconnect', () => {
      socket.broadcast.to(room).emit('userDisconnect', id);
    });
  });
});

server.listen(port, () => {
  console.log("Server running on port " + port);
});
```

3. Python Flask Server (Gesture Recognition API)
```bash
@app.route('/predict', methods=['POST'])
def predict():
    frames = request.files.getlist('frames[]')
    processed = process_frames(frames)       # RGB + optical flow
    prediction = model(processed)            # PyTorch forward pass
    predicted_class = torch.argmax(prediction).item()
    return jsonify({'class': int(predicted_class)})
```

## ▶️ Running the Project
1. Start the Python Flask Server
```bash
cd HandRecognitionPythonServer
python flaskServer.py
```

2. Start the Node.js Web Application
```bash
cd NodeWebapp
npm install
node server.js
```

Then visit:
```bash
http://localhost:3000
```

## 📊 Model Training Summary

Dataset: Northwestern Hand Gesture Dataset

Model: Two-stream CNN (RGB + Optical Flow)

Framework: PyTorch + OpenCV

Notebook: Training_the_network.ipynb

Best checkpoint: northwestern_classifier53.pt

## 🔮 Future Improvements

Replace optical flow with transformer-based motion encoding

Deploy on edge devices (Jetson Nano / Raspberry Pi)

Add gesture-based UI interactions in the webapp

Improve frame batching and reduce inference latency

## 👤 Author

Jorge Víctor Turriate Llallire
Master’s student in Mechatronics, Machine Vision & AI
Experience in Deep Learning, Computer Vision, and Robotics.
