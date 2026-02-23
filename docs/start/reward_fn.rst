Customize Your Reward Function
==============================

Author: `Yuyang Ding <https://yyding1.github.io>`_

This document illustrates different reward designs and their usage in verl.

We take mathematical reasoning tasks as an example.

Introduction
------------

As the primary source of supervision, reward modeling directly shapes the optimization objective, making it a critical component in achieving stable and effective training.

In verl, users can specify a custom reward function by configuring the following fields:

.. code-block:: yaml

  reward:
    custom_reward_function:
      path: /path/to/file.py
      name: function_name

Here, ``path`` refers to the file path of the Python module, and ``name`` denotes the function name.
verl will load the specified file as a module and invoke the corresponding function during reward computation.

This document will talk about the following reward usage scenarios:

- **Rule-based reward**
  
  - Simple string matching
  - CPU-intensive tasks

- **External API invocation**
  
  - Sandbox fusion
  - Model serving

- **Model-based reward**
  
  - Discriminative reward model
  - Generative reward model

- **Hybrid reward scenarios**
