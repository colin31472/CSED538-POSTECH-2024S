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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from new_scheduler import loss_informed_scheduler


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 0                         # random seed
  config.num_epochs = 100                 # number of epochs
  config.initial_lr = 0.01               # learning rate
  config.lr_scheduler_name = "ours"        # lr_scheuler : {"constant", "linear", "ours", etc}
  config.lr_scheduler = loss_informed_scheduler
  config.model_name = "ResNet18"          # name of model
  config.batch_size = 32                  # size of minibatch
  config.task   = "image"                 # image or language
  config.from_pretrained = False          # whether to load pretrained model or not
  config.new_technique_args = {             # if lr_scheduler is 'ours', add required arguments in this dictionary
    "name": "loss_informed_scheduler",
    "factors" : [8, 4, 2, 1, 0.5, 0.25, 0.125],
    "batch_size" : 32,
    "initial_lr" : 1
  }
  config.optimizer = "AdamW"
  config.save_name = "./checkpoints/loss_informed_scheduler_num7factor_cifar100"

  return config