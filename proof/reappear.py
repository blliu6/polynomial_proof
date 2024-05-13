def reappear(agent, env):
    episode_return = 0
    state, info = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = agent.take_action(state, env.action)
        next_state, reward, done, truncated, info = env.step(action)
        episode_return += reward
        state = next_state
    print(f'episode_return:{episode_return}')
