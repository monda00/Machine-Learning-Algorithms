import gym
from gym import wrappers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# 環境の生成
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './movie/dqn', force=True)
nb_actions = env.action_space.n

# モデルの定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# エージェントの設定
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# 学習
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

env.stats_recorder.save_complete()
env.stats_recorder.done = True

env = wrappers.Monitor(env, './movie/dqn/test', force=True)
# 5エピソードで学習したモデルをテスト
dqn.test(env, nb_episodes=5, visualize=True)
env.stats_recorder.save_complete()
env.stats_recorder.done = True
