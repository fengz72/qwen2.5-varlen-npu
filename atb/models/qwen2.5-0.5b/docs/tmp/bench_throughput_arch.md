# bench_throughput.cpp 架构详解

> 文件: `atb/bench_throughput.cpp`
> 功能: Qwen2.5-0.5B OM 模型多流吞吐性能测试

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        main()                                │
│  ACL 初始化 → 创建 Context → 预计算 cos/sin 表 → 循环 sweep  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    run_benchmark()                           │
│                                                              │
│  1. 创建 N 个 StreamContext (各自加载模型)                    │
│  2. Warmup                                                   │
│  3. 启动 Producer 线程 ──→ BlockingQueue ──→ N 个 Consumer   │
│  4. 等待所有请求完成                                          │
│  5. 统计报告                                                  │
└─────────────────────────────────────────────────────────────┘
```

**核心设计决策**:
- 每个 stream 加载**独立的模型实例**(`aclmdlLoadFromFile`),因为动态 shape 模型的 `ExecuteAsync` 不是线程安全的,共享 model_id 会导致 500002 错误
- 使用显式 `aclrtContext` + `aclrtSetCurrentContext`,而非隐式 device context,确保多线程 ACL 调用正确

## 二、分步讲解

### 1. 常量与配置 (L25-50)

```cpp
static const int    MAX_BATCH_SIZE     = 11;     // p99=10.9
static const int    MAX_SEQ_LEN        = 218;    // p99=218
static const int    MAX_TOTAL_TOKENS   = MAX_BATCH_SIZE * MAX_SEQ_LEN;  // 2398
```

定义了线上场景的边界值,用于预分配 device buffer 的最大尺寸。所有请求的实际数据都不能超过这些上限。

`BATCH_FIXED = true` 开关控制 batch_size 是固定 10 还是随机 `normal(9.8, 0.35)`。

### 2. 计时工具 (L52-61)

```cpp
using Clock = std::chrono::high_resolution_clock;
static inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}
```

用微秒精度计时,除以 1000 转 ms。高精度时钟用于精确测量 5-30ms 的推理时间。

### 3. CosSinTable (L63-135)

**目的**: 预计算 RoPE 的 cos/sin 表,运行时按 position_ids gather。

#### 3.1 init() — 预计算表

```cpp
void init(int max_seq_len, int head_dim = 64, double base = 1000000.0) {
    int half = dim / 2;  // 32
    for (int i = 0; i < half; i++) {
        inv_freq[i] = 1.0f / std::pow(base, (2.0 * i) / dim);
    }
    for (int pos = 0; pos < max_len; pos++) {
        for (int j = 0; j < half; j++) {
            float freq = pos * inv_freq[j];
            cos_table[pos * dim + j]        = std::cos(freq);
            cos_table[pos * dim + j + half] = std::cos(freq);  // 复制到后半
            sin_table[pos * dim + j]        = std::sin(freq);
            sin_table[pos * dim + j + half] = std::sin(freq);
        }
    }
}
```

生成 `[218, 64]` 的表,每个 position 对应 64 维的 cos/sin 值。前 32 维和后 32 维相同(Qwen2 的 RoPE 格式)。

#### 3.2 gather() — 按 position_ids 截取

```cpp
void gather(const std::vector<int64_t>& pos_ids, ...) {
    for (int i = 0; i < t; i++) {
        int p = pos_ids[i];
        for (int d = 0; d < dim; d++) {
            cos_out[i * dim + d] = float_to_fp16(cos_table[p * dim + d]);
        }
    }
}
```

varlen 场景下 position_ids = `[0,1,2,...,207, 0,1,2,...,207, ...]`,每条 seq 从 0 重启。gather 从表中按 position 查表,输出 `[1, T, 64]` 的 fp16 tensor。

#### 3.3 float_to_fp16() — 手写 fp16 转换

```cpp
static uint16_t float_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = (x >> 23) & 0xFF;
    uint32_t frac = x & 0x7FFFFF;
    // 按 IEEE 754 半精度规则转换...
}
```

不依赖 `__float2half` 内置函数,纯 C++ 实现 fp32→fp16 转换,保证可移植性。

### 4. Request (L137-154)

```cpp
struct Request {
    std::vector<int64_t> input_ids;           // [T] 全 0
    std::vector<int64_t> actual_seq_lengths;  // [N] 累积长度
    std::vector<uint16_t> cos_data;           // [1, T, 64] fp16
    std::vector<uint16_t> sin_data;           // [1, T, 64] fp16
    int total_tokens;
    int batch_size;
    TimePoint arrive_time;    // 请求到达 (prep 前)
    TimePoint enqueue_time;   // prep 完成, 入队
    TimePoint dequeue_time;   // consumer 取出
    TimePoint infer_start;    // ExecuteAsync 前
    TimePoint infer_end;      // SyncStream 后
};
```

5 个时间戳实现完整延迟拆分:
```
arrive → enqueue: Prep (gather cos/sin)
enqueue → dequeue: Queue Wait
dequeue → infer_start: H2D memcpy (含在 inference 内)
infer_start → infer_end: Inference
```

### 5. RequestGenerator (L156-220)

```cpp
Request generate(int req_id, TimePoint arrive_time) {
    // 1. 采样 batch_size (固定 10 或 normal(9.8, 0.35))
    // 2. 采样 seq_lens (lognormal, avg=150, p99=218)
    // 3. 构建 input_ids [T] (全 0, token 内容不影响性能)
    // 4. 构建 actual_seq_lengths [N] (累积)
    // 5. 构建 position_ids (每条 seq 从 0 重启)
    // 6. gather cos/sin
}
```

模拟线上请求分布:
- `batch_size`: `normal(9.8, 0.35)` 匹配线上 avg=9.8, p99=10.9
- `seq_len`: `exp(normal(4.997, 0.167))` 匹配线上 avg=150, p99=218

### 6. BlockingQueue (L222-275)

```cpp
class BlockingQueue {
    std::queue<Request*> q_;
    std::mutex mtx_;
    std::condition_variable cv_not_full_;   // 队列未满
    std::condition_variable cv_not_empty_;  // 队列非空
    size_t max_size_;
    bool closed_;
};
```

**有界阻塞队列**,实现 producer-consumer 模式:
- `push()`: 队列满时阻塞等待,有空位后入队并通知 consumer
- `pop()`: 队列空时阻塞等待(带 500ms 超时),有数据后出队并通知 producer
- `close()`: 关闭队列,唤醒所有等待线程

队列深度 = `streams × 4`,起到**背压**作用:producer 不会无限堆积请求。

### 7. LatencyStats (L277-379)

```cpp
struct LatencyStats {
    std::vector<double> e2e_prep_times;   // arrive → infer_end
    std::vector<double> prep_times;       // arrive → enqueue
    std::vector<double> queue_times;      // enqueue → dequeue
    std::vector<double> infer_times;      // infer_start → infer_end
    std::map<int, std::vector<double>> per_stream;
    std::mutex mtx;  // 多线程写入保护
};
```

#### 7.1 record() — 线程安全记录

```cpp
void record(const Request& req, int stream_id) {
    std::lock_guard<std::mutex> lock(mtx);
    e2e_prep_times.push_back(elapsed_ms(req.arrive_time, req.infer_end));
    prep_times.push_back(elapsed_ms(req.arrive_time, req.enqueue_time));
    // ...
}
```

每个 consumer 完成推理后调用,用 mutex 保护并发写入。

#### 7.2 percentile() — 百分位计算

```cpp
static double percentile(std::vector<double> data, double p) {
    std::sort(data.begin(), data.end());
    size_t idx = (size_t)(data.size() * p / 100.0);
    return data[idx];
}
```

排序后按索引取值,计算 p50/p90/p99。

#### 7.3 report() — 打印报告

输出格式:
```
  Latency (ms):
  E2E (incl prep)      avg=29.71  p50=29.57  p90=30.50  p99=31.74  max=32.53
    Prep               avg=0.01   p50=0.01   p90=0.01   p99=0.02   max=0.04
    Queue Wait         avg=24.75  ...
    Inference          avg=4.89   ...
```

### 8. StreamContext (L381-551)

**每个 stream 的 ACL 资源容器**,核心类。

#### 8.1 构造函数 — 定义输入规格

```cpp
StreamContext(int sid, const std::string& model_path, int device_id) {
    input_specs = {
        {ACL_INT64,   {MAX_BATCH_SIZE},                MAX_BATCH_SIZE * 8},          // actual_seq_lengths
        {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM}, MAX_TOTAL_TOKENS * HEAD_DIM * 2}, // cos
        {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM}, MAX_TOTAL_TOKENS * HEAD_DIM * 2}, // sin
        {ACL_INT64,   {MAX_TOTAL_TOKENS},              MAX_TOTAL_TOKENS * 8},        // input_ids
    };
    max_output_bytes = MAX_BATCH_SIZE * VOCAB_SIZE * 2;  // [N, vocab] fp16
}
```

预分配最大尺寸的 buffer,避免每次推理 malloc/free。

#### 8.2 init() — 初始化 ACL 资源

```cpp
int init() {
    // 1. 加载独立模型实例 (关键: 不共享 model_id)
    aclmdlLoadFromFile(model_path_, &model_id);
    aclmdlGetDesc(model_desc, model_id);

    // 2. 创建 stream
    aclrtCreateStream(&stream);

    // 3. 创建 input/output dataset, 预分配 device buffer
    for (auto& spec : input_specs) {
        aclrtMalloc(&ptr, spec.max_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclCreateDataBuffer(ptr, spec.max_bytes);
        aclmdlAddDatasetBuffer(input_dataset, buf);
    }
}
```

**为什么每个 stream 加载独立模型**: 动态 shape 模型的 `aclmdlSetDatasetTensorDesc` + `aclmdlExecuteAsync` 不是线程安全的。共享 model_id 会导致并发执行返回 500002 错误。

#### 8.3 set_inputs() — 设置动态输入

```cpp
int set_inputs(const Request& req) {
    InputData inputs[4] = {
        {req.actual_seq_lengths.data(), ..., ACL_INT64, {N}},
        {req.cos_data.data(), ..., ACL_FLOAT16, {1, T, 64}},
        {req.sin_data.data(), ..., ACL_FLOAT16, {1, T, 64}},
        {req.input_ids.data(), ..., ACL_INT64, {T}},
    };

    for (int i = 0; i < 4; i++) {
        // 1. H2D memcpy: host 数据 → device buffer
        aclrtMemcpy(input_buffers[i], input_max_sizes[i],
                    inputs[i].ptr, inputs[i].bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        // 2. 设置动态 tensor desc (告诉 OM 当前实际的 shape)
        aclTensorDesc* desc = aclCreateTensorDesc(dtype, ndim, shape, ACL_FORMAT_ND);
        aclmdlSetDatasetTensorDesc(input_dataset, desc, i);
        aclDestroyTensorDesc(desc);
    }
}
```

动态 shape 模型的关键: 每次推理前必须设置当前实际 shape,因为 device buffer 预分配了最大尺寸,但实际数据可能更小。

#### 8.4 execute() — 异步执行 + 同步等待

```cpp
int execute() {
    aclmdlExecuteAsync(model_id, input_dataset, output_dataset, stream);
    aclrtSynchronizeStream(stream);  // 阻塞等待完成
}
```

`ExecuteAsync` 提交任务到 stream,`SyncStream` 等待该 stream 上的所有任务完成。多 stream 之间可以并行执行。

#### 8.5 cleanup() — 资源释放

```cpp
void cleanup() {
    // 1. 释放 input/output dataset 中的 buffer
    // 2. 销毁 stream
    // 3. 销毁 model desc
    // 4. unload model
}
```

逆序释放,避免依赖问题。

### 9. run_benchmark() (L553-693)

核心编排函数,流程:

#### 9.1 创建 stream contexts

```cpp
for (int s = 0; s < num_streams; s++) {
    auto ctx = std::make_unique<StreamContext>(s, model_path, device_id);
    ctx->init();  // 加载模型 + 创建 stream + 预分配 buffer
    streams.push_back(std::move(ctx));
}
```

#### 9.2 Warmup

```cpp
for (int i = 0; i < warmup; i++) {
    Request req = gen.generate(i, Clock::now());
    streams[i % num_streams]->set_inputs(req);
    streams[i % num_streams]->execute();
}
```

预热 NPU,确保首次推理的编译/缓存开销不计入计时。

#### 9.3 启动 consumer 线程

```cpp
for (int s = 0; s < num_streams; s++) {
    consumers.emplace_back([&, s]() {
        aclrtSetCurrentContext(acl_ctx);  // 关键: 线程内设置 context
        while (true) {
            Request* req_ptr = queue.pop(500);  // 从队列取请求
            if (req_ptr == nullptr) {
                if (!running.load()) break;
                continue;
            }
            req_ptr->dequeue_time = Clock::now();
            ctx->set_inputs(*req_ptr);      // H2D + 设置动态 shape
            req_ptr->infer_start = Clock::now();
            ctx->execute();                  // ExecuteAsync + SyncStream
            req_ptr->infer_end = Clock::now();
            stats.record(*req_ptr, s);       // 记录延迟
            completed.fetch_add(1);
            delete req_ptr;                  // 释放请求
        }
    });
}
```

每个 consumer 绑定一个 stream,循环从队列取请求执行。

#### 9.4 启动 producer 线程

```cpp
TimePoint bench_start = Clock::now();
std::thread producer([&]() {
    aclrtSetCurrentContext(acl_ctx);
    RequestGenerator gen(42, cos_sin_table);
    for (int i = 0; i < total_requests; i++) {
        TimePoint arrive = Clock::now();
        Request* req = new Request(gen.generate(i, arrive));  // prep
        req->enqueue_time = Clock::now();
        queue.push(req);  // 队列满时阻塞 (背压)
    }
});
```

producer 尽快生成请求入队,队列满时阻塞等待。

#### 9.5 等待完成 + 收集结果

```cpp
producer.join();  // 等 producer 完成

while (completed.load() + errors.load() < total_requests) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
TimePoint bench_end = Clock::now();

running.store(false);
queue.close();
for (auto& t : consumers) t.join();
```

等待所有请求完成,然后停止 consumer 线程。

### 10. print_sweep_table() (L695-725)

```cpp
static void print_sweep_table(const std::vector<BenchResult>& results) {
    // 输出表格:
    // Streams | QPS | Token/s | Prep avg | E2E avg | E2E+prep p99 | Infer avg | Infer p99
}
```

sweep 多个 stream 数后,汇总对比表。

### 11. main() (L749-840)

```cpp
int main(int argc, char* argv[]) {
    // 1. 解析命令行参数
    // 2. aclInit + aclrtSetDevice + aclrtCreateContext
    // 3. 预计算 cos/sin 表
    // 4. 循环 sweep: run_benchmark(model_path, n, ...)
    // 5. 打印 sweep 对比表
    // 6. 清理: DestroyContext + ResetDevice + Finalize
}
```

## 三、数据流总结

```
Producer                    Queue                    Consumers (N)
─────────                   ─────                    ─────────────
arrive_time = now()
generate(req):
  sample batch_size
  sample seq_lens
  build input_ids
  build actual_seq_lengths
  build position_ids
  gather cos/sin ──→ prep
enqueue_time = now()
queue.push(req) ──────→ [req, req, ...] ──────→ req = queue.pop()
                                                dequeue_time = now()
                                                set_inputs(req):
                                                  H2D memcpy
                                                  set dynamic shape
                                                infer_start = now()
                                                execute():
                                                  ExecuteAsync
                                                  SyncStream
                                                infer_end = now()
                                                stats.record(req)
                                                delete req
```

## 四、关键设计要点

| 设计点 | 原因 |
|--------|------|
| 每 stream 独立模型实例 | 动态 shape 模型 ExecuteAsync 非线程安全 |
| 显式 aclrtContext | 多线程必须共享同一 context |
| 有界队列 (streams×4) | 背压控制,防止内存爆炸 |
| 预分配最大 buffer | 避免 malloc/free 开销影响计时 |
| 5 个时间戳 | 精确拆分 Prep/Queue/Infer 延迟 |
| Warmup | 排除首次编译/缓存开销 |

## 五、延迟拆分示意

```
时间轴: ────────────────────────────────────────────→

arrive    enqueue           dequeue    infer_start    infer_end
  │          │                  │           │            │
  ├─ Prep ──┤                  │           │            │
  │          ├── Queue Wait ───┤           │            │
  │          │                  ├─ H2D ────┤            │
  │          │                  │           ├── Infer ─┤
  │          │                  │           │            │
  ├──────── E2E (incl prep) ────────────────────────────┤

E2E (incl prep) = Prep + Queue Wait + H2D + Inference
```
