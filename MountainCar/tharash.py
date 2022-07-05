import gym
env = gym.make("MountainCarContinuous-v0")
observation = env.reset()
while 1:
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(reward)


  if done:
    observation = env.reset()
env.close()