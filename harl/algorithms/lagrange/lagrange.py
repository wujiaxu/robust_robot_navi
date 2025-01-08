# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Lagrange."""

from __future__ import annotations

from collections import deque

# from numpy import dtype
import torch

class Lagrange:
    """Lagrange multiplier for constrained optimization.
    
    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier

    Attributes:
        cost_limit: the cost limit  
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier
        _lagrangian_multiplier: the lagrangian multiplier
        lambda_range_projection: the projection function of the lagrangian multiplier
        lambda_optimizer: the optimizer of the lagrangian multiplier    
    """

    # pylint: disable-next=too-many-arguments
    # K_d= 0 or K_p = 0 covers the traditional Lagrangian method
    def __init__(
        self,
        lagrangian_multiplier_init: float,
        lagrangian_upper_bound: float,
        lagrangian_lower_bound: float,
        k_p: float = 0.0,
        k_i: float = 0.0003,
        k_d: float = 0.0,
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float =0.95,
        pid_d_delay: int = 10

    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""

        assert k_p>=0
        assert k_i>=0
        assert k_d>=0

        self.lagrangian_upper_bound: float = lagrangian_upper_bound
        self.lagrangian_lower_bound: float = lagrangian_lower_bound
        self.pid_Kp: float = k_p
        self.pid_Ki: float = k_i
        self.pid_Kd: float = k_d
        self.pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha

        init_value = max(lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier: float = init_value
        self.pid_i: float = init_value
        self.cost_ds = deque(maxlen=pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0
        
        
        # self.I = 0
        # self.prev_cost = 0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier.
        
        Returns:
            the lagrangian multiplier
        """
        return self._lagrangian_multiplier

    def update_lagrange_multiplier(self, ep_cost_avg: float, cost_limit: float) -> None:
        """Update the lagrangian multiplier.
        
        Args:
            Jc: the mean episode cost
            
        Returns:
            the loss of the lagrangian multiplier
        """
        # delta = Jc-cost_limit
        # derivative = max(0,Jc-self.prev_cost)
        # self.I = max(0,self.I+self.k_i*delta)
        # self._lagrangian_multiplier =  max(0,self.k_p*delta+self.I+self.k_d*derivative)
        # self._lagrangian_multiplier = min(self._lagrangian_multiplier,self.lagrangian_upper_bound)

        delta = float(ep_cost_avg - cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)   
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0., self._cost_d - self.cost_ds[0])
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
            self.pid_Kd * pid_d)
        self._lagrangian_multiplier = max(self.lagrangian_lower_bound, pid_o)
        self._lagrangian_multiplier = min(self._lagrangian_multiplier, self.lagrangian_upper_bound)

        self.cost_ds.append(self._cost_d)