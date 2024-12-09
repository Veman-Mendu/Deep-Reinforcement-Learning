# Challenge of a Non-Stationary Target

In Deep Reinforcement Learning step-by-step training takes place for the agent to learn the task. During this step-by-step training the model is trained at each step
with random batch of experiences pulled from the memory we store for the training. 
During the training in order to update the weigths of the model loss is calculated between the predicted Q-values of the states with respect to the action and target values calculated
based on the Equation
```markdown
target = reward + (discount_factor * max(model(newstates)) * (1 - done)) 
```
