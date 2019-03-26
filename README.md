# ExploreRL
Reinforcement Learning Algorithms implemented in Tensorflow 2.0 and PyTorch. Each algorithm is tested on openai gym environments. These tests can be found within each notebooks folder in their respective folders


## <ins>Model Based Methods<ins>
- [Policy Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration) [(code)](explorerl/ModelBased/PolicyIteration.py) [(example)](explorerl/ModelBased/notebooks/PolicyIteration.ipynb)
- [Value Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration) [(code)](explorerl/ModelBased/PolicyIteration.py) [(example)](explorerl/ModelBased/notebooks/ValueIteration.ipynb)

## <ins>Model Free Methods<ins>

### Value Based
- [Sarsa](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) [(code)](explorerl/SARSA/) [(example)](explorerl/SARSA/notebooks/Sarsa.ipynb)
- [QLearning](https://en.wikipedia.org/wiki/Q-learning) [(code)](explorerl/QLearning/) [(example)](explorerl/QLearning/notebooks/QLearning.ipynb)
- [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) [(code)](explorerl/DQN/) [(example)](explorerl/DQN/notebooks/DQN.ipynb)

### Policy Gradient
- [REINFORCE](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) [(code)](explorerl/REINFORCE/) [(example)](explorerl/REINFORCE/notebooks/REINFORCE.ipynb)