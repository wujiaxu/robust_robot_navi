from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,

class DrlVoVecEnv(SubprocVecEnv):

    def __init__(self,crowd_model,env_fns,start_method="forkserver"):

        super(DrlVoVecEnv,self).__init__(env_fns,start_method)

        self.crowd_model = crowd_model