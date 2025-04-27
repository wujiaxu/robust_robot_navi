from gym.envs.registration import register

# Register drl_nav env 
register(
  id='drl_vo_env-v0',
  entry_point='turtlebot_gym.envs:DRLNavEnv'
  )

register(
  id='drl_vo_env-v1',
  entry_point='turtlebot_gym.envs:DRLNavEnvForVec'
  )
