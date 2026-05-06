 # DeepSeek-V4-Flash-Base
#  python3 -m sglang.bench_serving \
#     --backend vllm \
#     --host 127.0.0.1 --port 30010 \
#     --model /hf/DeepSeek-V4-Flash-Base \
#     --tokenizer /hf/DeepSeek-V4-Flash-Base \
#     --dataset-name random \
#     --random-input-len 1024 --random-output-len 1024 \
#     --random-range-ratio 0.8 \
#     --num-prompts 40 --max-concurrency 4 \
#     --request-rate inf \
#     --warmup-requests 8 \
#     --seed 1 \
#     --profile \
#     --profile-num-steps 5


python3 -m sglang.bench_serving \
    --backend vllm \
    --host 127.0.0.1 --port 30014 \
    --model /hf/DeepSeek-V4-Pro \
    --tokenizer /hf/DeepSeek-V4-Pro \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --num-prompts 40 --max-concurrency 4 \
    --request-rate inf \
    --warmup-requests 8 \
    --seed 1 \
    --profile \
    --profile-num-steps 5