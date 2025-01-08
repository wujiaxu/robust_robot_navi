import torch
from torch import nn
from harl.models.base.rnn import RNNLayer
from harl.models.base.mlp import MLPLayer
from harl.utils.envs_tools import check
from harl.utils.models_tools import init, get_init_method
import torch.nn.functional as F

class BehaviorDiscriminator(nn.Module):
    """
    Defines the discriminator network and the training objective for the
    discriminator. Through the Habitat Baselines auxiliary loss registry, this
    is automatically added to the policy class and the loss is computed in the
    policy update.
    """

    def __init__(
        self,
        input_size, 
        behavior_latent_dim,
        hidden_sizes,
        initialization_method,
        device
    ):
        super().__init__()
        self.input_dim = input_size
        self.behavior_latent_dim = behavior_latent_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPLayer(input_size,hidden_sizes,initialization_method,"relu")
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        self.discrim_output = init_(nn.Linear(hidden_sizes[-1], behavior_latent_dim))

        self.to(device)

    def pred_logits(self, policy_features):
        return self.discrim_output(policy_features)

    def forward(self,obs,rnn_states, actions,active_mask=None):
        label = obs[...,-self.behavior_latent_dim:]
        label = check(label).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        rnn_states = rnn_states.reshape(actions.shape[0],-1)
        if active_mask is not None:
            active_mask = check(active_mask).to(**self.tpdv).squeeze(-1)

        behav_ids = torch.argmax(label, -1)
        input_discrim = torch.cat([rnn_states,actions],dim=-1)
        discrim_features = self.base(input_discrim)
        pred_logits = self.pred_logits(discrim_features)
        loss = F.cross_entropy(pred_logits, behav_ids,reduction="none") 
        # if torch.any(loss>0):
        #     print(pred_logits)
        #     raise ValueError
        return loss if active_mask is None else loss*active_mask