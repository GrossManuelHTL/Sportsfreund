import "reflect-metadata"; 
import express from "express";
import raspyfitRouter from "./routes/router";
import { Database } from "./database/Database";

const app = express();
const PORT = 3000;

app.use(express.json());
app.use("/sportsfreund", raspyfitRouter);


Database.initialize().then(() => {
  app.listen(PORT, () => {
    console.log(`local server runs on http://localhost:${PORT}`);
  });
});
