# python data_process.py --local_dir /mnt/hdfs/resources/fp_bench/

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=/mnt/hdfs/resources/fp_bench/processbench_full.parquet \
    data.prompt_key=prompt \
    data.batch_size=2048 \
    data.n_samples=1 \
    data.output_path=/mnt/hdfs/resources/fp_bench/processbench_full_qwen3-8b-ins_greedy-32768.parquet \
    model.path=Qwen/Qwen3-8B \
    rollout.temperature=0.0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=4096 \
    rollout.response_length=32768 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.95 \
    rollout.max_num_batched_tokens=65536

# python evaluate.py --parquet_path /mnt/hdfs/resources/fp_bench/processbench_full_qwen2_5-7b-ins_greedy.parquet
