import { Router } from "express";
import { Database } from "../database/Database";
import { Exercise } from "../database/entities/Exercise.entity";

const router = Router();

router.post("/exercise", async (req, res) => {
  try {
    const repo = Database.getRepo(Exercise);
    const newExercise = repo.create(req.body);
    const result = await repo.save(newExercise);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get("/exercises", async (_req, res) => {
  try {
    const repo = Database.getRepo(Exercise);
    const all = await repo.find();
    res.json(all);
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get("/session", (req, res)=>{
  console.log("heast wos isn des");
})

export default router;
