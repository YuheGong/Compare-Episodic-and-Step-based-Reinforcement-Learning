import argparse
from utils.env import env_maker, env_save
from utils.logger import logging
from utils.model import model_building, model_learn
from utils.yaml import write_yaml, read_yaml
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG


def step_based(algo: str, env_Id: str):
    file_name =  algo +".yml"
    data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path

    # choose the algorithm according to the config file
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


    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'])
    test_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    # CustomPolicy = CustomActorCriticPolicy
    model = model_building(data, env)

    #model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
    #             # model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
    #             tensorboard_log=data['path'],
    #             seed=3,
    #             learning_rate=data["algo_params"]['learning_rate'],
    #             batch_size=data["algo_params"]['batch_size'],
    #             n_steps=data["algo_params"]['n_steps'])
    try:
        test_env_path = data['path'] + "/eval/"
        model_learn(data, model, test_env, test_env_path)
        # print("test_env_path",test_env_path)
        #model.learn(total_timesteps=int(data['algo_params']['total_timesteps']), eval_freq=2048, n_eval_episodes=8,
        #            eval_log_path=test_env_path, eval_env=test_env)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training interrupt, save the model and config file to ' + data["path"])
    else:
        # save the model
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])

def episodic():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str,
                    help="the algorithm")

    parser.add_argument("--env_id", type=str,
                    help="the environment")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')


    algo = args.algo
    env_id = args.env_id
    #print(algo, env_id)

    STEP_BASED = ["ppo", "sac"]
    EPISODIC = ["cmaes"]
    if algo in STEP_BASED:
        step_based(algo, env_id)
    elif algo in EPISODIC:
        episodic()
    else:
        print("the algorithm (--algo) is false or not implemented")






    '''
    if args.model_path:  # overwrite is model_path is given
        args.config_file = os.path.join(args.model_path, 'config.yml')
    
    # configure experiment
    exp = Experiment(args.config_file)
    
    if args.model_path:
        exp.load(args.model_path)
        exp.test_learned()
    else:              # run experiment
        exp.run()
        exp.test_learned()
    
    '''
