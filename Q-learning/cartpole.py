import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

total_reward = 0

for step in range(1000):
    pole_angle = observation[2]
    pole_angle_velocity = observation[3]

    # 比只看角度更聪明一点
    if pole_angle + 0.5 * pole_angle_velocity < 0:
        action = 0
    else:
        action = 1

    current_angle = observation[2]
    current_angle_v = observation[3]

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print(
        f"step={step+1}, "
        f"angle={current_angle:.4f}, "
        f"angle_v={current_angle_v:.4f}, "
        f"action={action}, "
        f"reward={reward}"
    )

    time.sleep(0.05)

    if terminated or truncated:
        print(f"该回合结束, total_reward: {total_reward}")
        break

env.close()