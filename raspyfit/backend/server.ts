import express from 'express';
const server = express();
const port = 3000;

server.get('/', (req, res) => {
  res.send('Simulierte Raspberry Pi API');
});

server.listen(port, () => {
  console.log(`Server läuft auf http://localhost:${port}`);
});