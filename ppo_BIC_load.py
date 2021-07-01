import gym
import os
import time
import alr_envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from alr_envs.deep_mind.ball_in_cup.ball_in_cup import DeepMindBallInCup
from alr_envs.deep_mind.env_wrapper import DMEnvWrapper
import matplotlib.pyplot as plt
from dm_control import suite

path = "./logs/ppo/DeepMindBallInCupDense-v0_9"

def make_env(rank, seed=0):
    def _init():
        env = suite.load(domain_name="ball_in_cup", task_name="catch")
        base_env = DeepMindBallInCup(env)
        env = DMEnvWrapper(base_env, render_size=(480, 480))
        # env = wrappers.Monitor(env)#, path, force=True)
        return env

    return _init


#base_env = DeepMindBallInCup
#env = DMEnvWrapper(base_env, render_size = (480, 480))

n_cpu = 1
#env = gym.make('DeepMindBallInDense-v0_1')
stats_path = os.path.join(path, "PPO.pkl")
env = DummyVecEnv(env_fns=[make_env(i) for i in range(n_cpu)])
env = VecNormalize.load(stats_path, env)#, norm_obs=True)#, norm_reward=True,clip_obs=10.)

model_path = os.path.join(path, "PPO.zip")
model = PPO.load(model_path)

obs = env.reset()

for i in range(220):
    #time.sleep(0.01)
    action, _states = model.predict(obs)#, deterministic = True)
    #print("predict_action", action)
    #if i <= 15:
        #action = -action

    obs, rewards, dones, info = env.step(action)
    #env.render(mode="rgb_array")
    #print("outside_action", action)

    video = env.render(mode="rgb_array")  # (height, width, camera_id=0)
    # for i in range(max_frame):
    img = plt.imshow(video)
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.draw()
env.close()   
 

