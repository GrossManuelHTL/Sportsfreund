import { Router } from "express";
import { Database } from "../database/Database";
import { Exercise } from "../database/entities/Exercise.entity";
import { Session } from "../database/entities/Session.entity"
import {SessionExercise} from "../database/entities/SessionExercise.entity";

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

router.post("/session", async (req, res)=>{
  try{
    const repo = Database.getRepo(Session);
    const newSession = repo.create(req.body)
    const result = await repo.save(newSession);
    res.json(result);
  }catch (error){
    res.status(500).json("Server error"+ error);
  }
})

router.patch("/session/:id/endtime", async (_req, _res) => {
  const { id } = _req.params;

  try {
    await Database.initialize();

    const sessionRepo = Database.getRepo(Session);
    const session = await sessionRepo.findOneBy({ id: parseInt(id) });

    if (!session) {
      _res.status(404).send("Session not found");
    }

    session!.endtime = new Date();
    await sessionRepo.save(session!);

    _res.status(200).json("endtime succesfully updated");
  } catch (error: any) {
    _res.status(500).json("Database Error" + error);
  }
});

router.post("/sessions/exercise", async (req, res) =>{
  try{
    const repo = Database.getRepo(SessionExercise);
    const newSessionExercise = repo.create(req.body)
    const result = await repo.save(newSessionExercise);
    res.json(result);
  }catch (error){
    res.status(500).json("Server error"+ error);
  }
})

export default router;
