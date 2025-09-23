# Sportsfreund

## Project Description

Sportsfreund is an inclusive, multi-platform fitness companion designed for everyone. The app provides guided exercises, interactive feedback, adaptive training plans, and social motivation through gamification. Users can access Sportsfreund via a native phone app (for now only Android) or a responsive web app.

## Goals

- Deliver an accessible, multi-modal fitness experience that serves users of all abilities.
- Encourage daily healthy habits with gentle reminders, progress tracking, and gamified rewards.
- Offer fast, private, and reliable access to workouts and personalized coaching.

## Key Features

- Cross-platform access: native mobile apps (React Native) and a responsive web app (React).
- Personalization: onboarding questionnaire, adjustable difficulty levels, adaptive exercise recommendations.
- Workout library: short guided sessions, full routines, and custom workouts with audio and visual cues.
- Scheduling & reminders: calendar integration, push notifications, and daily prompts.
- Gamification:
  - Daily check-ins and reminders if a user hasn't completed their exercise for the day.
  - Streak tracking with visible streaks on the dashboard (e.g., "3-day streak").
  - Badges and achievements for milestones (first week completed, 30-day streak, consistency awards).
  - Optional social features: leaderboards, friend challenges, and group streaks (privacy-first, opt-in).
- Progress & analytics: session history, streak history, weekly/monthly summaries, and simple charts.

## Accessibility & Inclusion

- Designed for sighted users and users with visual impairments: multimodal feedback (audio, haptics, visuals).
- Configurable interaction modes: voice-first, touch-first, or hybrid.
- Localization and cultural sensitivity in language, measurements, and content.
- Privacy and consent: clear data controls, opt-in analytics, and GDPR-compliant data handling.

## Technical Overview

- Frontend: React (web) and React Native or Flutter (mobile) for shared UI and consistent experiences.
- Backend: Django backend
- Notifications: push notifications for mobile, web push for browsers, and email fallback.
- Offline-first basics: local caching for workouts and progress, sync when online.
- Integrations: calendar, health platforms (optional), social sharing (opt-in).

## Security & Privacy

- Minimize collected data; store only what is necessary for functionality.
- Encrypt PII at rest and in transit.
- Provide account deletion, data export, and privacy settings in-app.

## Success Metrics

- Daily active users (DAU) and retention rates.
- Average streak length and percentage of users achieving multi-week streaks.
- Accessibility compliance score and feedback from users with disabilities.
- Net promoter score (NPS) and feature adoption rates.

## Roadmap (high level)

1. Core cross-platform app with accessible UI and core workout library.
2. Scheduling, reminders, and basic gamification (streaks, badges).
3. Advanced personalization, social features (opt-in), and integrations.
4. Analytics, continuous accessibility improvements, and localization.

Sportsfreund aims to make fitness engaging and reachable for everyone by combining inclusive design with motivating gamification.
