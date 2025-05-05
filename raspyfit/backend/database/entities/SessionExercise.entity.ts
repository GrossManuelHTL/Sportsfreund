import { Entity, Column, PrimaryColumn, ManyToOne, Index } from 'typeorm';
import { Session } from './Session.entity';
import { Exercise } from './Exercise.entity';

@Entity('Session_Exercise')
export class SessionExercise {
    @PrimaryColumn({ type: 'integer' })
    sessionID!: number;

    @PrimaryColumn({ type: 'integer' })
    exerciseID!: number;

    @Column({ type: 'text', nullable: true })
    feedback!: string;

    @Column({ type: 'integer' })
    repetitions!: number;

    @Column({ type: 'real' })
    score!: number;

    @Index('idx_session_id')
    @ManyToOne(() => Session, (session) => session.sessionExercises)
    session!: Session;

    @Index('idx_exercise_id')
    @ManyToOne(() => Exercise, (exercise) => exercise.sessionExercises)
    exercise!: Exercise;
}