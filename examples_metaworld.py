import alr_envs
from alr_envs.utils.make_env_helpers import make_env



def example_dmc(env_id="fish-swim", seed=1, iterations=1000, render=True, episodic = False):
    """
    Example for running a MetaWorld based env in the step based setting.
    The env_id has to be specified as `task_name-v2`. V1 versions are not supported and we always
    return the observable goal version.
    All tasks can be found here: https://arxiv.org/pdf/1910.10897.pdf or https://meta-world.github.io/

    Args:
        env_id: `task_name-v2`
        seed: seed for deterministic behaviour (TODO: currently not working due to an issue in MetaWorld code)
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    env = make_env(env_id, seed)
    rewards = 0
    obs = env.reset()
    print("observation shape:", env.observation_space.shape)
    action = []
    #print("action shape:", env.action_space.shape)

    if episodic:
        env.render(mode="meta")

    for i in range(iterations):

        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward
        action.append(ac)

        if render:
            # THIS NEEDS TO BE SET TO FALSE FOR NOW, BECAUSE THE INTERFACE FOR RENDERING IS DIFFERENT TO BASIC GYM
            # TODO: Remove this, when Metaworld fixes its interface.
            env.render(False)



        if done:
            print(env_id, rewards)
            rewards = 0
            obs = env.reset()

    env.close()
    del env


if __name__ == '__main__':
    # Disclaimer: MetaWorld environments require the seed to be specified in the beginning.
    # Adjusting it afterwards with env.seed() is not recommended as it may not affect the underlying behavior.

    # For rendering it might be necessary to specify your OpenGL installation
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    render = True

    # # Standard DMC Suite tasks
    example_dmc("button-press-v2", seed=10, iterations=500, render=render)

    # MP + MetaWorld hybrid task provided in the our framework
    #example_dmc("ButtonPressDetPMP-v2", seed=10, iterations=1, render=render, episodic=True)

    # Custom MetaWorld task
    #example_custom_dmc_and_mp(seed=10, iterations=1, render=render)
