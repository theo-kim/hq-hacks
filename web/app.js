var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

app.get('/about', function(req, res){
  res.sendFile(__dirname + '/about.html');
});

app.get('/demo', function(req, res){
  res.sendFile(__dirname + '/demo.html');
});

io.on('connection', function(socket){
  socket.on('disconnect', function(){
  });

  socket.on('init', function() {
    io.emit('init', { for : 'everyone' })
  });

  socket.on('info', function(answer) {
    console.log('answer', answer)
    io.emit('info', answer)
  });

  socket.on('answer', function(answer) {
    io.emit('answer', answer)
  });
});

http.listen(process.env.PORT || 3000, function(){
  console.log('listening on *:8080');
});