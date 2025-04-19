import torch
import gymnasium as gym

from torch.optim import Adam

class PPO:
	def __init__(self, environment, network_class , **hyperparameters):
    assert isinstance(env.observation_space, gym.spaces.Box), "Observation space must be of type Box"
    assert isinstance(env.action_space, gym.spaces.Box), "Action space must be of type Box"

    self.env = env
		self.observation_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.shape[0]

    self.actor = network_class(self.observation_dim, self.action_dim) # Step 1          
    self.critic = network_class(self.observation_dim, 1)

    self.hyperparameter_init(hyperparameters)

		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

  def hyperparameter_init(self, hyperparameters):
  	self.TIMESTEPS_PER_BATCH = 4800                 
  	self.TIMESTEPS_PER_EPISODE = 1600  
  	self.UPDATES_PER_ITERATION = 5  
  	self.lr = 0.005                     
  	self.gamma = 0.95                  
  	self.clip = 0.2    

  def learn(self, total_timesteps)
  	timesteps_completed = 0
  	iter_count = 0 
  	while timesteps_completed < total_timesteps:
  		batch_observations, batch_actions, batch_log_probs, batch_reward_to_gos, batch_episode_lengths = self.rollout()  # Step 3
  
    	t_so_far += np.sum(batch_lens)

      V, _ = self.evaluate(batch_obs, batch_acts) # Step 5
      A_k = batch_reward_to_gos - V.detach() 

      for _ in range(self.UPDATES_PER_ITERATION):    
      	V, curr_log_probs = self.evaluate(batch_observations, batch_actions)
      
      	ratios = torch.exp(curr_log_probs - batch_log_probs)
      
      	#Calculating surrogate loss r_\theta
      
      	surr1 = ratios * A_k
      	surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
      
      	actor_loss = (-torch.min(surr1, surr2)).mean() # Negative because adam minimizes, and we want to maximize.
      	critic_loss = torch.nn.MSELoss()(V, batch_reward_to_gos)

	# Next - we calculate the gradients for both the networks

	self.actor_optimiser.zero_grad()
	actor_loss.backward(retain_graph=True)
	self.actor_optimiser.step()

	self.critic_optimiser.zero_grad()
	critic_loss.backward()
	self.critic_optimiser.step()

  	iter_count +=1

  def rollout(self):
  	# Batch data that gets returned
  	batch_observations = [] 
  	batch_actions = []
  	batch_log_probs = []
  	batch_rewards = [] 
  	batch_reward_to_gos = []
  	batch_episode_lengths = []
  
  	episode_rewards=[] # Keeps track of rewards per episode, get's cleared after every episode.
  
  	timesteps = 0
  
  	while t < self.TIMESTEPS_PER_BATCH: # 
  		episode_rewards = [] 
  
  		# Resetting the environment after each batch.
  		observation, _ = self.env.reset()
  		done = False
  
  		for episode_timestep in range(self.TIMESTEPS_PER_EPISODE):
  			# Rendering the environment
  			if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
  					self.env.render()
  			# Incrementing The Timestep
  			timesteps += 1
  
  			# Now - we'll track the observations in this batch,
  			batch_observations.append(observation)
  
  			# We'll be using our get_action function to calculate the action & log_prob
  			
  			action, log_prob = self.get_action(observation)
  			observation, reward, terminated, truncated, _ = self.env.step(action)
  
  			done = truncated | terminated
  
  			# Tracking the recent rewards, & action, probs
  			episode_rewards.append(reward)
  			batch_actions.append(action)
  			batch_log_probs.append(log_prob)
  			
  			if done:
  				break
  
  		# Tracking lengths
  		batch_episode_lengths.append(ep_t + 1)
  		batch_rewards.append(episode_rewards)
  
  	# Reshaping Data Into Tensors
  	batch_observations = torch.tensor(batch_observations, dtype=torch.float)
  	batch_actions = torch.tensor(batch_actions, dtype=torch.float)
  	batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
  	batch_reward_to_gos = self.compute_reward_to_gos(batch_rewards)     # STEP 4   
  	
  	return batch_obs, batch_acts, batch_log_probs, batch_reward_to_gos, batch_lens

  def compute_reward_to_gos(self, batch_rewards):
  	batch_reward_to_gos = []
  
  	for episode_rewards in reversed(batch_rewards):
  		discounted_reward = 0
  		for reward in reversed(episode_rewards):
  			discounted_reward = reward + discounted_reward * self.gamma
  			batch_reward_to_gos.insert(0, discounted_reward)
  
  	batch_reward_to_gos = torch.tensor(batch_reward_to_gos, dtype=torch.float)
  
  	return batch_reward_to_gos

  def get_action(self, observation)
    mean = self.actor(observation)
    
    dist = torch.distributions.MultivariateNormal(mean, self.cov_mat) # Defines a distribution
    
    
    action = dist.sample() # Samples action
    
    
    log_prob = dist.log_prob(action) # Calculates log probability for action
    
    return action.detach().numpy(), log_prob.detach()

  def evaluate(self, batch_observations, batch_actions)
  		V = self.critic(batch_observations).squeeze()
  
  		mean = self.actor(batch_observations)
  		dist = MultivariateNormal(mean, self.cov_mat)
  		log_probs = dist.log_prob(batch_actions)
  
  		return V, log_probs
