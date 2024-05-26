const express= require('express');
const app= express();

const server = require('http').Server(app);
const io = require('socket.io')(server);
const port = process.env.PORT || 3000;
const {ExpressPeerServer} = require('peer')
const peer = ExpressPeerServer(server , {
  debug:true
});

// Static room ID
const staticRoomId = 'static-room-id';

//Routes
const videoRouter= require('./routes/videoRouter');


app.use('/peerjs', peer);
app.set('views','views');
app.set('view engine', 'ejs');

//Calling the public folder
app.use(express.static('public'));

//Handling the get request to root
app.get('/', (req, res)=>{
    res.redirect(`/meeting/home`);
});


//Video routes
app.use('/meeting', videoRouter);


io.on('connection' , (socket)=>{
    socket.on('newUser' , (id, room)=>{
        socket.join(room);
        socket.broadcast.to(room).emit("userJoined" , id);
        socket.on('disconnect', ()=>{
            socket.broadcast.to(room).emit('userDisconnect', id);
        })

    });
});

//Listening the server
server.listen(port, ()=>{
    console.log("Server running on port "+ port);
})