import gym

env = gym.make('CartPole-v0')

#Restart State
#Returns the State
s =env.reset()

print(s)

box = env.observation_space

print(box)

env.action_space

done = False

while not done:

    observation,reward, done, _ = env.step(env.action_space.sample())