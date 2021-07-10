env = VecNormalize(make_vec_env(env_name), training=True, norm_obs=True, norm_reward=True)
env_test = VecNormalize(make_vec_env(env_name), training=False, norm_obs=True, norm_reward=False)
model = PPO("MlpPolicy", env, verbose=1, create_eval_env=True, device='cpu',
            tensorboard_log=tb_path,
            policy_kwargs={'net_arch': [dict(pi=[cw_config.params.net_work_const, cw_config.params.net_work_const],
                                             vf=[cw_config.params.net_work_const, cw_config.params.net_work_const])]},
            seed=c_seed,
            learning_rate=cw_config.params.l_r_val)
model_save_path = cw_config['_rep_log_path'] + '/'# + exp_name
start_time = time.time()
log_name = 'PPO_' + str(cw_config.params.l_r_val) + '_' + str(cw_config.params.net_work_const)
model.learn(total_timesteps=cw_config.params.total_timesteps, n_eval_episodes=cw_config.params.n_eval_episodes,
            log_interval=1,
            eval_freq=cw_config.params.eval_freq,
            tb_log_name=env_name + log_name, eval_log_path=tb_path, eval_env=env_test)
model.save(tb_path + '/' + 'last_model')
env.save(tb_path + '/' + 'env_normalize.pkl')