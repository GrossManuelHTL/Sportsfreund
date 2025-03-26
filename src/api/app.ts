import express from 'express';
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Simulierte Raspberry Pi API');
});

app.listen(port, () => {
  console.log(`Server läuft auf http://localhost:${port}`);
});