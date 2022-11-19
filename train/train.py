import numpy as np
from keras.optimizers import Adam

from classical_model.model import build_model, build_agent
from environment.protein_folding_environment import ProteinFoldingEnvironment

protein_sequence = "HPHPPHHPHPPHPHHPPHPH"
env = ProteinFoldingEnvironment(protein_sequence,len(protein_sequence))
states = env.observation_space.shape
actions = env.action_space.n
model = build_model(states, actions)
model.summary()
dqn =build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))
_ = dqn.test(env, nb_episodes=15, visualize=True)
dqn.save_weights('dqn_weights.h5f', overwrite=True)