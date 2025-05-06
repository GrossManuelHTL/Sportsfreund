import {Entity, Column, PrimaryColumn, OneToMany, CreateDateColumn, PrimaryGeneratedColumn} from 'typeorm';
import { SessionExercise } from './SessionExercise.entity';

@Entity('Session')
export class Session {
  @PrimaryGeneratedColumn({ type: 'integer' })
  sessionID!: number;

  @CreateDateColumn({ type: 'date' })
  start!: Date;

  @CreateDateColumn({ type: 'date' })
  end!: Date;

  @Column({ type: 'real', default: 0 })
  totalScore!: number;

  @OneToMany(() => SessionExercise, (sessionExercise) => sessionExercise.session)
  sessionExercises!: SessionExercise[];
}