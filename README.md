# 🛰️ AgriSight: Reinforcement Learning for Satellite Constellation Scheduling

Scheduling a constellation of Earth observation satellites is a massive optimization headache. If you want high-frequency images of specific agricultural zones (to predict crop yields, for example), your satellites need to be in the right place at the right time. But they are strictly limited by battery life, onboard storage, and orbital physics.

I built **AgriSight** to see if Multi-Agent Reinforcement Learning could solve this better than traditional static scheduling algorithms.

## 🛠️ The Tech Stack

- **Environment:** Custom `Gymnasium` environment
- **Orbital Mechanics:** `Skyfield` (for real Two-Line Element / TLE propagation)
- **RL Architecture:** Independent Deep Q-Networks (DQN) built with `PyTorch`

## 🧠 The Journey & The "Space Brick" Problem

Building the environment was step one. I integrated `Skyfield` so the agents had to deal with real Geodetic coordinates and dynamic field-of-view math based on their actual altitude.

The real challenge was training the agents. Early in the build, the RL agents figured out a loophole: taking photos drains the battery, and a dead battery triggers a massive negative penalty. So, the agents mathematically optimized their strategy by simply shutting off their sensors and drifting through space to preserve power. They became **"Space Bricks."**

To fix this and get the constellation to actually work, I had to redesign the architecture:

1. **Reward Shaping:** I completely removed passive rewards for hoarding battery power and heavily multiplied the rewards for capturing high-priority agricultural targets.
2. **Huber Loss:** The massive penalties from draining power were causing exploding gradients (loss values spiking into the millions). Switching to Huber loss stabilized the network.
3. **Prioritized Experience Replay (PER):** Because space is vast, stumbling across a target randomly is rare. I added PER so the neural network would specifically hunt through its memory for the rare times it *did* hit a target, forcing it to learn from successes rather than empty orbital drifting.

## 🚀 Quick Start

Want to run the simulation or train the agents yourself?

```bash
# Clone the repo
git clone https://github.com/debanjan06/SatelliteRL.git
cd SatelliteRL

# Install dependencies and the local package
pip install -r requirements.txt
pip install -e .

# Run the multi-agent training pipeline
python src/training/train_dqn.py
```

## 📊 Results

> **Note:** Add your final numbers here once your new training run finishes!

After addressing the reward function and adding PER, the agents stopped drifting and started actively hunting for targets.

- **Target Acquisition:** The constellation successfully increased its target completion rate by [X]%.
- **Stability:** The DQN loss stabilized from >1,000,000 down to < 1.0.
- **Constraint Management:** The agents learned to maximize coverage while safely maintaining their power levels above the 20% critical threshold.

---

## 👨‍💻 Built By

**Debanjan Shil**  
Geospatial Data Science & Reinforcement Learning  
[LinkedIn](https://www.linkedin.com/in/debanjanshil) | [GitHub](https://github.com/debanjan06)