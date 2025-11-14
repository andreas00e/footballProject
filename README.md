# 🏈 NFL Big Data Bowl 2026 — Prediction
https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
Participants are asked to predict player movements during the ball-in-air phase. The NFL provides tracking data up to the moment the quarterback releases the ball, 
including Next Gen Stats data. Additionally, participants are given the targeted offensive player and the pass’s landing location.

📌 Overview

This repository contains my work for the NFL Big Data Bowl 2026.
The task:
Given player tracking data before the ball is snapped, predict the future trajectories of a subset of players during the ball-in-air phase.

The project explores multiple modeling paradigms including:

2D positional encodings for spatial awareness

Transformer-based sequence models

Dynamic graph neural networks to capture player-to-player interactions

Hybrid fusion architectures combining sequence + graph reasoning

🎯 Problem Description

Before the snap, we observe:

Player positions (x, y)

Player velocities and speeds

Orientation, acceleration, roles, and team alignment

Frame-by-frame movement at ~10 Hz

We must use only pre-snap motion to forecast in-air movement once the QB throws the ball.

This is a structured spatiotemporal forecasting problem with relational dependencies (teammates, opponents, formations).

🧠 Modeling Approach
1. 2D Positional Encoding for Each Frame

To help the model understand field geometry, each frame receives:

A sinusoidal positional encoding on the continuous (x, y) field coordinates

Encodings concatenated to player states before being passed into the model

This helps the network learn:

spatial patterns (formations, spacing)

standard NFL field layout

differences between tight/loose formations, hash marks, etc.

2. Frame-Level Embeddings → Temporal Transformer

Each frame is embedded into a unified vector, then passed into a transformer encoder:
