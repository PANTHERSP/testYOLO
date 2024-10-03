const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors'); // เพิ่มการเรียกใช้งาน cors

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: '*', // หรือระบุโดเมนที่ต้องการอนุญาต
    methods: ['GET', 'POST']
  }
});

// ใช้ cors กับ express
app.use(cors());

io.on('connection', (socket) => {
  console.log('WebSocket client connected');

  socket.on('video_frame', (frame) => {
    io.emit('video_frame', frame);
  });

  socket.on('disconnect', () => {
    console.log('WebSocket client disconnected');
  });
});

server.listen(5501, () => {
  console.log('Server is running on port 5501');
});
