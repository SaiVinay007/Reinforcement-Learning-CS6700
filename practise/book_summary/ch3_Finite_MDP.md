# Chapter 3
# Finite Markov Decision Processes

- MDPs are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards.

- This problem involves evaluative feedback, as in bandits, but also an associative aspect—choosing different actions in different situations.

- Whereas in bandit problems we estimated the value q ∗ (a) of each action a, in MDPs we estimate the value q ∗ (s, a) of each action a in each state s, or we estimate the value v ∗ (s) of each state given optimal action selections. These state-dependent quantities are essential to accurately assigning credit for long-term consequences to individual action
selections.


### 3.1 The Agent–Environment Interface
