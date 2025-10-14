# MONTE CARLO CONTROL ALGORITHM

## AIM
Develop a Python program to find the optimal policy for a given RL environment using Monte Carlo Control.

## PROBLEM STATEMENT
Use the Monte Carlo Control algorithm to find the optimal policy for the FrozenLake environment from episode trajectories.

## ALGORITHM OVERVIEW
1. **Initialization:**  
   - Set the number of states (`ns`) and actions (`na`).  
   - Create a Q-table (`ns × na`) to store action values.  
   - Define decaying schedules for learning rate (`alpha`) and exploration rate (`epsilon`).  
   - Use a discount factor (`gamma`) and epsilon-greedy action selection.  

2. **Training Loop:**  
   - For each episode, generate a trajectory of `(state, action, reward, next_state, done)`.  
   - Compute return `G` for each state-action pair.  
   - Update Q-values: `Q(s,a) ← Q(s,a) + α * (G - Q(s,a))`.  
   - Track Q-values and greedy policy per episode.  

3. **Policy Extraction:**  
   - Compute state-value function `V` as `V(s) = max_a Q(s,a)`.  
   - Define learned policy `pi(s) = argmax_a Q(s,a)`.  

## MONTE CARLO CONTROL FUNCTION
```python
from tqdm import tqdm

def mc_control(env, gamma=1.0,
               init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    ns, na = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((ns, na), dtype=np.float64)
    Q_track = np.zeros((n_episodes, ns, na), dtype=np.float64)
    pi_track = []

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(na)

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((ns, na), dtype=bool)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * np.array([x[2] for x in trajectory[t:]]))
            Q[state][action] += alphas[e] * (G - Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])
    return Q, V, pi, Q_track, pi_track
```
# OUTPUT 
<img width="1186" height="376" alt="image" src="https://github.com/user-attachments/assets/6d0ed031-e28c-4a38-8348-c80bd3f94520" />

<img width="942" height="193" alt="image" src="https://github.com/user-attachments/assets/2a9b58a2-d4c6-4220-b750-b699e1cfa949" />


# RESULT

The Python program successfully finds the optimal policy for the FrozenLake environment using Monte Carlo Control.

