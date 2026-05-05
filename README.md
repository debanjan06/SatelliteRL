# SatelliteRL: Multi-Agent Reinforcement Learning for Satellite Constellation Scheduling

Scheduling a constellation of Earth observation satellites is a massive optimization headache. If you want high-frequency images of specific geographical zones, your satellites need to be in the right place at the right time. But they are strictly limited by battery life, onboard storage capacity, and the relentless physics of their orbits.

**SatelliteRL** is a custom multi-agent environment and training pipeline designed to solve this exact bottleneck. It tests whether an Independent Deep Q-Network (DQN) can autonomously coordinate a decentralized constellation better than traditional static heuristic algorithms.

## The Tech Stack

- **Environment:** Custom `Gymnasium` environment (230-dimensional continuous state space)
- **Orbital Mechanics:** `Skyfield` (Processing real Two-Line Element / TLE data)
- **RL Architecture:** Independent Deep Q-Networks (DQN) built with `PyTorch`

## The Architecture & The "Space Brick" Anomaly

Building the environment was step one. I integrated `Skyfield` so the agents had to navigate real Geodetic coordinates and dynamic field-of-view trigonometry based on their actual altitude.

The real engineering challenge was training the agents. Early in the build, the RL agents figured out a mathematical loophole: operating sensors drains the battery, and a dead battery triggers a massive negative reward penalty. So, the agents optimized their strategy by simply shutting off their sensors and drifting through space to preserve power. They became **"Space Bricks."**

To force the constellation to actually observe the Earth, I overhauled the architecture:

1. **Aggressive Reward Shaping:** I removed passive rewards for hoarding battery power and heavily multiplied the rewards for capturing high-priority targets, forcing the agents to balance risk vs. reward.
2. **Prioritized Experience Replay (PER):** Because space is vast, stumbling across a target randomly is rare. I implemented PER so the neural network would specifically hunt through its memory for the rare times it *did* hit a target, forcing it to learn from successes rather than empty orbital drifting.
3. **Huber Loss:** The massive penalties from draining power were causing exploding gradients. Switching to Huber loss stabilized the network.

## Quick Start

Want to run the simulation, test the baselines, or train the agents yourself?

```bash
# Clone the repo
git clone https://github.com/debanjan06/SatelliteRL.git
cd SatelliteRL

# Install dependencies and the local package
pip install -r requirements.txt
pip install -e .

# Run the multi-agent training pipeline
python src/training/train_dqn.py

# Benchmark the DQN against Random & Greedy algorithms
python src/simulation/run_basic_sim.py
```

## Benchmark Results

> **Note:** Final metrics pending complete GPU training cycle. The benchmark suite `run_basic_sim.py` currently pits the multi-agent DQN against greedy and random baselines inside the Skyfield physics engine.

---

## Built By

**Debanjan Shil**  
Geospatial Data Science & Reinforcement Learning  
[LinkedIn](https://www.linkedin.com/in/debanjanshil) | [GitHub](https://github.com/debanjan06)# SatelliteRL: Multi-Agent Reinforcement Learning for Satellite Constellation Scheduling

Scheduling a constellation of Earth observation satellites is a massive optimization headache. If you want high-frequency images of specific geographical zones, your satellites need to be in the right place at the right time. But they are strictly limited by battery life, onboard storage capacity, and the relentless physics of their orbits.

**SatelliteRL** is a custom multi-agent environment and training pipeline designed to solve this exact bottleneck. It tests whether an Independent Deep Q-Network (DQN) can autonomously coordinate a decentralized constellation better than traditional static heuristic algorithms.

## The Tech Stack

- **Environment:** Custom `Gymnasium` environment (230-dimensional continuous state space)
- **Orbital Mechanics:** `Skyfield` (Processing real Two-Line Element / TLE data)
- **RL Architecture:** Independent Deep Q-Networks (DQN) built with `PyTorch`

## The Architecture & The "Space Brick" Anomaly

Building the environment was step one. I integrated `Skyfield` so the agents had to navigate real Geodetic coordinates and dynamic field-of-view trigonometry based on their actual altitude.

The real engineering challenge was training the agents. Early in the build, the RL agents figured out a mathematical loophole: operating sensors drains the battery, and a dead battery triggers a massive negative reward penalty. So, the agents optimized their strategy by simply shutting off their sensors and drifting through space to preserve power. They became **"Space Bricks."**

To force the constellation to actually observe the Earth, I overhauled the architecture:

1. **Aggressive Reward Shaping:** I removed passive rewards for hoarding battery power and heavily multiplied the rewards for capturing high-priority targets, forcing the agents to balance risk vs. reward.
2. **Prioritized Experience Replay (PER):** Because space is vast, stumbling across a target randomly is rare. I implemented PER so the neural network would specifically hunt through its memory for the rare times it *did* hit a target, forcing it to learn from successes rather than empty orbital drifting.
3. **Huber Loss:** The massive penalties from draining power were causing exploding gradients. Switching to Huber loss stabilized the network.

## Quick Start

Want to run the simulation, test the baselines, or train the agents yourself?

```bash
# Clone the repo
git clone https://github.com/debanjan06/SatelliteRL.git
cd SatelliteRL

# Install dependencies and the local package
pip install -r requirements.txt
pip install -e .

# Run the multi-agent training pipeline
python src/training/train_dqn.py

# Benchmark the DQN against Random & Greedy algorithms
python src/simulation/run_basic_sim.py
```

## Benchmark Results

> **Note:** Final metrics pending complete GPU training cycle. The benchmark suite `run_basic_sim.py` currently pits the multi-agent DQN against greedy and random baselines inside the Skyfield physics engine.

---

## Built By

**Debanjan Shil**  
Geospatial Data Science & Reinforcement Learning  
[LinkedIn](https://www.linkedin.com/in/debanjanshil) | [GitHub](https://github.com/debanjan06)
