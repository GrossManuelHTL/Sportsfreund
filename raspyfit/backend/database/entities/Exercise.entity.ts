import { Entity, Column, PrimaryGeneratedColumn, OneToMany } from 'typeorm';
import { SessionExercise } from './SessionExercise.entity';

@Entity('Exercise')
export class Exercise {
  @PrimaryGeneratedColumn({ type: 'integer' })
  exerciseID!: number;

  @Column({ type: 'text' })
  name!: string;

  @Column({ type: 'text' })
  description!: string;

  @Column({ type: 'integer', nullable: true })
  difficulty!: number;

  @Column({ type: 'text', nullable: true })
  comment!: string;

  @OneToMany(() => SessionExercise, (sessionExercise) => sessionExercise.exercise)
  sessionExercises!: SessionExercise[];
}