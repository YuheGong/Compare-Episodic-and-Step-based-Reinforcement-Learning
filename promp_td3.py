import argparse
from utils.env import env_maker, env_save
from utils.logger import logging
from utils.model import model_building, model_learn, cmaes_model_training
from utils.yaml import write_yaml, read_yaml
import numpy as np
import gym
import cma
from torch.utils.tensorboard import SummaryWriter
from utils.csv import csv_save


def step_based(algo: str, env_id: str, seed=None):
    file_name = algo +".yml"
    data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path
    data['seed'] = seed

    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'])
    eval_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    model = model_building(data, env, seed)

    # csv file path
    data["path_in"] = data["path"] + '/' + data['algorithm'].upper() + '_1'
    data["path_out"] = data["path"] + '/data.csv'

    try:
        eval_env_path = data['path'] + "/eval/"
        model_learn(data, model, eval_env, eval_env_path)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training interrupt, save the model and config file to ' + data["path"])
    else:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])

def episodic(algo, env_id, stop_cri, seed=None):
    file_name = algo + ".yml"
    data = read_yaml(file_name)[env_id]
    env_name = data["env_params"]["env_name"]
    if 'Meta' in env_id:
        from alr_envs.utils.make_env_helpers import make_env
        env = make_env(env_name, seed)
    else:
        env = gym.make(env_name[2:-1], seed=seed)

    params = data["algo_params"]['x_init'] * np.random.rand(data["algo_params"]["dimension"])
    ALGOS = {
        'cmaes': cma,
    }
    if data["algorithm"] == "cmaes":
        algorithm = ALGOS[data["algorithm"]].CMAEvolutionStrategy(x0=params, sigma0=data["algo_params"]["sigma0"], inopts={"popsize": data["algo_params"]["popsize"]})
    #else:
        #algorithm = GradientDescent()

    # logging
    path = "alr_envs:" + env_id
    path = logging(path, algo)
    log_writer = SummaryWriter(path)

    t = 0
    opts = []
    success = False
    success_mean = []
    success_full = []


    gamma = 0.01

    # initialize the solutions
    import torch
    from torch.distributions.multivariate_normal import MultivariateNormal
    mean = torch.ones(25)
    #cov1 = torch.eye(3)
    #cov = torch.Tensor([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    cov = torch.eye(25)
    mean.requires_grad = True
    cov.requires_grad = True
    mean.retain_grad()
    cov.retain_grad()
    #print("cov", mean)
    #param = cov
    #cov = torch.stack([cov1, cov2], 0)


    try:
        while t < data["algo_params"]["iteration"]:
            fitness = []
            print("----------iter {} -----------".format(t))
            # solutions = np.vstack(algorithm.ask())

            solutions = []
            log_probs = []
            for i in range(data["algo_params"]["popsize"]):
                #print("mean", mean)
                #print("cov", cov)
                distrib = MultivariateNormal(loc=mean, covariance_matrix=torch.tril(cov))
                d_1 = distrib.rsample((1,))  # .resize(5,1)
                log_1 = distrib.log_prob(d_1)
                # print("d_1", d_1.T)
                # print('log_1', torch.exp(log_1))
                # assert 1==1231
                if not log_1.requires_grad:
                    log_1.requires_grad = True
                log_prob = torch.mean(log_1)
                #print("log_prob", log_prob.grad)
                log_prob.retain_grad()
                # v = torch.cat((d_1,d_2, d_3, d_4, d_5),dim =0)
                # v.requires_grad = True
                solution = d_1.T.detach().numpy()
                #print("d_1", d_1)
                solutions.append(solution)
                log_probs.append(log_prob)
            solutions = np.array(solutions)

            q_all = []
            for i in range(len(solutions)):
                env.reset()
                #print("solutions[i]", solutions[i])
                _, reward, done, infos = env.step(solutions[i])
                if "DeepMind" in env_id:
                    success_full.append(env.env.success)
                print('reward', -reward)
                #fitness.append(-reward)


                y = 0
                ## calculate the td3 loss
                ## as we have no neural network here, we only use the reward at each step
                for i in reversed(range(infos['step_rewards'].shape[0])):
                    if i == infos['step_rewards'].shape[0] - 1:
                        y += infos['step_rewards'][i]
                    else:
                        y += infos['step_rewards'][i] + gamma * y
                    # print("yyyyy",y)
                # assert 1==123
                #y = np.array(0 - y)
                #print("took")
                q_all.append(y)
                #print("q_all", q_all)
                #q_func = torch.from_numpy(y)
                #q_func.requires_grad = True
                #q_all.cat((q,q_func),0)

                # print("q",q_func)
                # print("infos['step_rewards']",infos['step_rewards'])
                # assert 1==123

            #print("q_al", q_all)
            #print("log_prob", log_probs)
            env.reset()
            test = distrib.mean
            test = test.detach().numpy()
            #print("test", test)

            _, opt, __, ___ = env.step(test)

            np.save(path + "/algo_mean.npy", test)
            print("opt", -opt)
            opts.append(opt)
            log_writer.add_scalar("iteration/reward", opt, t)

            mul = []
            for i in range(len(q_all)):
                #a = q_all[i] * log_probs[i]
                mul.append(-q_all[i] * torch.exp(log_probs[i]))
                #log_probs[i]= torch.exp(log_probs[i])
            #print("multi", mul)
            #log_probs = torch.exp(log_probs)

            #x_n_mu =
            cov_in = torch.inverse(cov)
            for i in range(len(mul)):
                #mul[i].zero)grad
                mul[i].retain_grad()
                mul[i].backward(retain_graph=True)
                mul[i].retain_grad()
                #print(i, log_probs[i].grad)
                gra = log_probs[i].grad
                #log_probs[i].retain_grad()
                #log_probs[i].backward()

                #print("log_prob", solutions[i])

                mu = mean
                #print("mu",mu)
                c = gra * torch.ones(25)
                c = c
                #print("c",c)
                d = c * (d_1 - mu)

                #a = (d.resize(1, 5))
                #b = (d.resize(5, 1))
                #print(b * a)
                #x_n = log_probs[i] ^ 2 * a * b
                if i == 0:
                    x_n_mu = d*d.T
                    x_n_sig = cov_in * d * d.T * cov_in
                else:
                    x_n_mu += d*d.T
                    x_n_sig += cov_in * d * d.T * cov_in
            #print("x_n_mu", x_n_mu)
                #assert 1==12311
                #mul[i].backward()
                #print("log_prob", log_probs[i].grad)
            N = len(mul)

            mu_gra = - N / 2 * cov_in + 1/2 * cov_in * x_n_mu * cov_in
            sigma_gra =  cov_in * N/2 - x_n_sig

            mean = mean - 0.01 * mu_gra
            mean = torch.mean(mean, dim = 1)
            cov = cov - 0.01 * sigma_gra
            #print("mu_gra", sigma_gra)
            #mean = mean + log_probs[i]
            #assert 1==123
            #optimizer = torch.optim.adam([mean, cov])
            #env.reset()

            #algorithm.tell(solutions, fitness)
            #_, opt, __, ___ = env.step(algorithm.mean)

            #np.save(path + "/algo_mean.npy", algorithm.mean)
            #print("opt", -opt)
            #opts.append(opt)

            t += 1
            if "DeepMind" in env_id:
                success_mean.append(env.env.success)
                if success_mean[-1]:
                    success_rate = 1
                else:
                    success_rate = 0

                b = 0
                for i in range(len(success_full)):
                    if success_full[i]:
                        b += 1
                success_rate_full = b / len(success_full)
                success_full = []

            if "DeepMind" in env_id:
                log_writer.add_scalar("iteration/success_rate_full", success_rate_full, t)
                log_writer.add_scalar("iteration/success_rate", success_rate, t)
                log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t)
                log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t)
            #log_writer.add_scalar("iteration/reward", opt, t)

            # log_writer.add_scalar("iteration/dist_vec", env.env.dist_vec, t)
            #for i in range(len(algorithm.mean)):
            #    log_writer.add_scalar(f"algorithm_params/mean[{i}]", algorithm.mean[i], t)
            ##    log_writer.add_scalar(f"algorithm_params/covariance_matrix_mean[{i}]", np.mean(algorithm.C[i]), t)
            #    log_writer.add_scalar(f"algorithm_params/covariance_matrix_variance[{i}]", np.var(algorithm.C[i]), t)

    except KeyboardInterrupt:
        data["path_in"] = path
        data["path_out"] = path + '/data.csv'
        csv_save(data)
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training interrupt, save the model to ' + path)
    else:
        data["path_in"] = path
        data["path_out"] = path + '/data.csv'
        csv_save(data)
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training Finish, save the model to ' + path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--stop_cri", type=str, help="whether you set up stop criterion or not")
    parser.add_argument("--seed", type=int, help="seed for training")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    stop_cri = args.stop_cri
    STEP_BASED = ["ppo", "sac", "ddpg"]
    #print("algo", algo)
    EPISODIC = ["dmp", "promp"]
    if algo in STEP_BASED:
        step_based(algo, env_id, seed=args.seed)
    elif algo in EPISODIC:
        episodic(algo, env_id, stop_cri, seed=args.seed)
    else:
        print("the algorithm " + algo + " is false or not implemented")

