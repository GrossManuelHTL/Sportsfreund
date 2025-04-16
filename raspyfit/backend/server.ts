import "reflect-metadata"; 
import express from "express";
import testRouter from "./routes/router";
import { Database } from "./database/Database";

const app = express();
const PORT = 3000;

app.use(express.json());
app.use("/test", testRouter);


Database.initialize().then(() => {
  app.listen(PORT, () => {
    console.log(`local server runs on http://localhost:${PORT}`);
  });
});
