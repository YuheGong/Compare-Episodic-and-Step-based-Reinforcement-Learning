
from utils.callback import ALRBallInACupCallback,DMbicCallback
from utils.custom import CustomActorCriticPolicy
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy


def model_building(data, env, seed=None):
    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[data['algorithm']]

    POLICY = {
        'MlpPolicy': MlpPolicy,
        'CustomActorCriticPolicy': CustomActorCriticPolicy
    }
    if "special_policy" in data['algo_params']:
        policy = POLICY[data['algo_params']['special_policy']]
    else:
        policy = data['algo_params']['policy']

    model = ALGO(policy, env, verbose=1, create_eval_env=True,
                 # model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
                 tensorboard_log=data['path'],
                 seed=seed,
                 learning_rate=data["algo_params"]['learning_rate'],
                 batch_size=data["algo_params"]['batch_size'],
                 n_steps=data["algo_params"]['n_steps'])
    return model


def model_learn(data, model, test_env, test_env_path):
    # choose the tensorboard callback function according to the environment wrapper
    CALLBACKS = {
            'ALRBallInACupCallback': ALRBallInACupCallback(),
            'DMbicCallback': DMbicCallback()
        }
    if 'special_callback' in data['algo_params']:
        callback = CALLBACKS[data['algo_params']['special_callback']]
    else:
        callback = None

    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(test_env, best_model_save_path=test_env_path,  n_eval_episodes=10,
                                 log_path=test_env_path, eval_freq=500,
                                 deterministic=False, render=False)

    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']), callback=eval_callback)

                #, eval_freq=500, n_eval_episodes=10, eval_log_path=test_env_path, eval_env=test_env)