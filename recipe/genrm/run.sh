set -x

# python3 -m verl.trainer.main_ppo \
python3 -m recipe.genrm.genrm_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/hdfs/resources/datasets/GSM8K-Processed/train.parquet \
    data.val_files=/mnt/hdfs/resources/datasets/GSM8K-Processed/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/hdfs/resources/models/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-debug' \
    trainer.experiment_name='qwen2_5_3b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.resume_mode='disable'