import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load results
if os.path.exists('results/episode_rewards.npy'):
    rewards = np.load('results/episode_rewards.npy')
    completions = np.load('results/episode_completions.npy')
    
    print("🚀 SatelliteRL Training Results Summary")
    print("="*50)
    print(f"📊 Episodes trained: {len(rewards)}")
    print(f"🏆 Best reward: {max(rewards):.2f}")
    print(f"📈 Final average reward (last 25): {np.mean(rewards[-25:]):.2f}")
    print(f"🎯 Final average completions (last 25): {np.mean(completions[-25:]):.1f}")
    print(f"📉 Improvement: {np.mean(rewards[-25:]) - np.mean(rewards[:25]):.2f}")
    
    # Quick plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.7)
    window = 25
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'red', linewidth=2)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(completions, alpha=0.7, color='green')
    if len(completions) >= window:
        moving_avg = np.convolve(completions, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(completions)), moving_avg, 'darkgreen', linewidth=2)
    plt.title('Target Completions')
    plt.xlabel('Episode')
    plt.ylabel('Targets Completed')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quick_summary.png', dpi=150)
    plt.show()
    
    print(f"📈 Summary plot saved to results/quick_summary.png")
    
else:
    print("❌ No training results found. Make sure training completed.")