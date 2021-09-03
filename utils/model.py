import numpy as np
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
    eval_callback = EvalCallback(test_env, best_model_save_path=test_env_path,  n_eval_episodes=data['eval_env']['n_eval_episode'],
                                 log_path=test_env_path, eval_freq=data['eval_env']['eval_freq'],
                                 deterministic=False, render=False)

    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']), callback=eval_callback)
                #, eval_freq=500, n_eval_episodes=10, eval_log_path=test_env_path, eval_env=test_env)


def cmaes_model_training(algorithm, env, success_full, success_mean, opt_full, fitness, path, log_writer, opts, t):
    print("----------iter {} -----------".format(t))
    solutions = np.vstack(algorithm.ask())
    for i in range(len(solutions)):
        # print(i, solutions[i])
        #print(env.step)
        #assert 1==238
        env.reset()
        _, reward, done, ___ = env.step(solutions[i])
        #print("done_in_model", done)
        success_full.append(env.env.success)
        # env.reset()
        print('reward', -reward)

        opt_full.append(reward)
        fitness.append(-reward)
        #env.reset()
    #print("self.sp.cmean", algorithm.C)
    #assert 1==237
    algorithm.tell(solutions, fitness)
    #print("mean2", algorithm.C)
    _, opt, __, ___ = env.step(algorithm.mean)


    success_mean.append(env.env.success)
    if success_mean:
        success = True
    env.reset()

    np.save(path + "/algo_mean.npy", algorithm.mean)
    log_writer.add_scalar("iteration/reward", opt, t)
    log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t)
    log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t)
    for i in range(len(algorithm.mean)):
        log_writer.add_scalar(f"algorithm_params/mean[{i}]", algorithm.mean[i], t)
        #print(i, algorithm.C[i])
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_mean[{i}]", np.mean(algorithm.C[i]), t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_variance[{i}]", np.var(algorithm.C[i]), t)

    fitness = []
    opts.append(opt)
    # opt_full.append(reward)
    t += 1

    if t % 1 == 0:
        a = 0
        b = 0

        # print(len(opts))
        for i in range(len(success_mean)):

            if success_mean[i]:
                a += 1
        success_rate = a / len(success_mean)
        success_mean = []
        for i in range(len(success_full)):
            if success_full[i]:
                b += 1
        success_rate_full = b / len(success_full)
        success_full = []

        # print("success_full_rate", success_rate_full)
        log_writer.add_scalar("iteration/success_rate_full", success_rate_full, t)
        log_writer.add_scalar("iteration/success_rate", success_rate, t)
    return algorithm, env, success_full, success_mean, opt_full, fitness, path, log_writer, opts, t