import { Entity, Column, PrimaryColumn, OneToMany, CreateDateColumn } from 'typeorm';
import { SessionExercise } from './SessionExercise.entity';

@Entity('Session')
export class Session {
  @PrimaryColumn({ type: 'text' })
  sessionID!: string;

  @CreateDateColumn({ type: 'date' })
  start!: Date;

  @CreateDateColumn({ type: 'date' })
  end!: Date;

  @Column({ type: 'real', default: 0 })
  totalScore!: number;

  @Column({ type: 'text', nullable: true })
  comment!: string;

  @OneToMany(() => SessionExercise, (sessionExercise) => sessionExercise.session)
  sessionExercises!: SessionExercise[];
}