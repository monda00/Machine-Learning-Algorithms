import gym

# 環境の生成
env = gym.make('CartPole-v0')

for i_episode in range(20):
    # 環境を初期化してobsersavationを取得
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # アクションの決定
        action = env.action_space.sample()
        # アクション後のデータを取得
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
