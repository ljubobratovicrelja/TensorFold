## matrix_bench: tools/dflash2/run/matrix.jsonl
- measured rows: 27 (all warm: one unmeasured pass of the same prompts before each cell); error rows: 0; cold-prefix rows: 0; floor: 3 reps per cell

### mtp-131k | path `mtp` | http://127.0.0.1:8099/v1
- server: /v1/models id `qwen3.8-flash-next` at http://127.0.0.1:8099/v1; request model `qwen3.8-flash-next`; one path per run, so every row below is the server as started

#### fill 0 tokens
- server counted 76 prompt tokens
- lanes=1 | decode 107.2 tok/s | aggregate 104.8 tok/s | ttft 0.064 s | ttft max 0.064 s | prefill 0.063 s | prompt 76 tok | cached 69 tok (91%) | n=3 reps
- lanes=2 | decode 96.0 tok/s | aggregate 185.7 tok/s | ttft 0.098 s | ttft max 0.132 s | prefill 0.064 s | prompt 76 tok | cached 138 tok (91%) | n=3 reps
- lanes=4 | decode 84.6 tok/s | aggregate 321.5 tok/s | ttft 0.168 s | ttft max 0.272 s | prefill 0.099 s | prompt 76 tok | cached 276 tok (91%) | n=3 reps

#### fill 20000 tokens
- server counted 19995 prompt tokens
- lanes=1 | decode 106.9 tok/s | aggregate 103.3 tok/s | ttft 0.092 s | ttft max 0.092 s | prefill 0.088 s | prompt 19995 tok | cached 19988 tok (100%) | n=3 reps
- lanes=2 | decode 84.6 tok/s | aggregate 156.6 tok/s | ttft 0.182 s | ttft max 0.275 s | prefill 0.097 s | prompt 19995 tok | cached 39976 tok (100%) | n=3 reps
- lanes=4 | decode 67.0 tok/s | aggregate 237.6 tok/s | ttft 0.351 s | ttft max 0.653 s | prefill 0.182 s | prompt 19995 tok | cached 79952 tok (100%) | n=3 reps

#### fill 100000 tokens
- server counted 100178 prompt tokens
- lanes=1 | decode 102.5 tok/s | aggregate 96.1 tok/s | ttft 0.172 s | ttft max 0.172 s | prefill 0.155 s | prompt 100178 tok | cached 100171 tok (100%) | n=3 reps
- lanes=2 | decode 78.8 tok/s | aggregate 140.2 tok/s | ttft 0.316 s | ttft max 0.472 s | prefill 0.185 s | prompt 100178 tok | cached 200342 tok (100%) | n=3 reps
- lanes=4 | decode 59.6 tok/s | aggregate 204.6 tok/s | ttft 0.537 s | ttft max 0.960 s | prefill 0.282 s | prompt 100178 tok | cached 400684 tok (100%) | n=3 reps

- coverage: fill 0: 3/3 lanes; fill 20000: 3/3 lanes; fill 100000: 3/3 lanes; 9 of 9 cells measured
