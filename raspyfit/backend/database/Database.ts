import "reflect-metadata";
import { DataSource } from "typeorm";
import { Exercise } from "./entities/Exercise.entity";
import { Session } from "./entities/Session.entity";
import { SessionExercise } from "./entities/SessionExercise.entity";

export class Database {
  private static AppDataSource = new DataSource({
    type: "sqlite",
    database: "raspyfit_db.sqlite",
    synchronize: true,
    logging: false,
    entities: [Exercise, Session, SessionExercise],
  });

  static async initialize() {
    if (!this.AppDataSource.isInitialized) {
      await this.AppDataSource.initialize();
      console.log("Database connected!");
    }
  }

  static getRepo(entity: any) {
    return this.AppDataSource.getRepository(entity);
  }
}
