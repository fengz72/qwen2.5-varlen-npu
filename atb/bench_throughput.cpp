#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <random>
#include <cmath>
#include <map>

#include "acl/acl.h"

// ============================================================================
// Constants
// ============================================================================

#define ACL_CHECK(ret, msg) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            std::cerr << "[ERROR] " << msg << ", ret=" << ret << std::endl; \
            return ret; \
        } \
    } while (0)

static const int    VOCAB_SIZE         = 151936;
static const int    HEAD_DIM           = 64;
static const double ROPE_BASE          = 1000000.0;
static const int    MAX_BATCH_SIZE     = 11;     // p99=10.9
static const int    MAX_SEQ_LEN        = 218;    // p99=218
static const int    MAX_TOTAL_TOKENS   = MAX_BATCH_SIZE * MAX_SEQ_LEN;  // 2398

// Distribution params (matching production)
static const double BATCH_AVG   = 9.8;
static const double BATCH_STD   = 0.35;
static const bool   BATCH_FIXED = true;   // fixed batch_size=10 for controlled test
static const int    BATCH_FIXED_VAL = 10;
static const double SEQ_LOG_MEAN = 4.997;
static const double SEQ_LOG_STD  = 0.167;

// ============================================================================
// Timing helpers
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

static inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Cos/Sin table (precomputed on host)
// ============================================================================

struct CosSinTable {
    std::vector<float> inv_freq;       // [32]
    std::vector<float> cos_table;      // [max_len * 64]
    std::vector<float> sin_table;      // [max_len * 64]
    int max_len;
    int dim;

    void init(int max_seq_len, int head_dim = HEAD_DIM, double base = ROPE_BASE) {
        max_len = max_seq_len;
        dim = head_dim;
        int half = dim / 2;
        inv_freq.resize(half);
        for (int i = 0; i < half; i++) {
            inv_freq[i] = 1.0f / std::pow(base, (2.0 * i) / dim);
        }
        cos_table.resize(max_len * dim);
        sin_table.resize(max_len * dim);
        for (int pos = 0; pos < max_len; pos++) {
            for (int j = 0; j < half; j++) {
                float freq = pos * inv_freq[j];
                cos_table[pos * dim + j]         = std::cos(freq);
                cos_table[pos * dim + j + half]  = std::cos(freq);
                sin_table[pos * dim + j]         = std::sin(freq);
                sin_table[pos * dim + j + half]  = std::sin(freq);
            }
        }
    }

    // Gather cos/sin for given position_ids, output fp16
    void gather(const std::vector<int64_t>& pos_ids,
                std::vector<uint16_t>& cos_out,
                std::vector<uint16_t>& sin_out) const {
        int t = pos_ids.size();
        cos_out.resize(t * dim);
        sin_out.resize(t * dim);
        for (int i = 0; i < t; i++) {
            int p = pos_ids[i];
            for (int d = 0; d < dim; d++) {
                float c = cos_table[p * dim + d];
                float s = sin_table[p * dim + d];
                // fp16 conversion
                cos_out[i * dim + d] = float_to_fp16(c);
                sin_out[i * dim + d] = float_to_fp16(s);
            }
        }
    }

    static uint16_t float_to_fp16(float f) {
        uint32_t x = *reinterpret_cast<uint32_t*>(&f);
        uint16_t h;
        uint32_t sign = (x >> 31) & 0x1;
        int32_t  exp  = (x >> 23) & 0xFF;
        uint32_t frac = x & 0x7FFFFF;
        if (exp == 0xFF) {
            h = (sign << 15) | 0x7C00 | (frac ? 0x200 : 0);
        } else if (exp >= 142) {
            h = (sign << 15) | 0x7C00;
        } else if (exp >= 113) {
            h = (sign << 15) | ((exp - 112) << 10) | (frac >> 13);
        } else if (exp >= 103) {
            int shift = 113 - exp;
            h = (sign << 15) | (frac >> (shift + 13));
            if ((frac >> (shift + 12)) & 1) h++;
        } else {
            h = (sign << 15);
        }
        return h;
    }
};

// ============================================================================
// Request
// ============================================================================

struct Request {
    int req_id;
    std::vector<int64_t> input_ids;           // [T]
    std::vector<int64_t> actual_seq_lengths;  // [N]
    std::vector<uint16_t> cos_data;           // [1, T, 64] fp16
    std::vector<uint16_t> sin_data;           // [1, T, 64] fp16
    int total_tokens;
    int batch_size;
    TimePoint arrive_time;    // request arrives (before prep)
    TimePoint enqueue_time;   // prep done, enqueued
    TimePoint dequeue_time;
    TimePoint infer_start;
    TimePoint infer_end;
};

// Random request generator matching production distributions
class RequestGenerator {
public:
    RequestGenerator(unsigned seed, const CosSinTable& table)
        : rng_(seed), table_(table),
          batch_dist_(BATCH_AVG, BATCH_STD) {}

    Request generate(int req_id, TimePoint arrive_time) {
        Request req;
        req.req_id = req_id;
        req.arrive_time = arrive_time;

        // batch_size: fixed or normal distribution
        int bs;
        if (BATCH_FIXED) {
            bs = BATCH_FIXED_VAL;
        } else {
            bs = (int)std::round(batch_dist_(rng_));
            bs = std::max(1, std::min(MAX_BATCH_SIZE, bs));
        }
        req.batch_size = bs;

        // seq_lens: lognormal(avg=150, p99=218)
        std::vector<int> seq_lens(bs);
        int total = 0;
        for (int i = 0; i < bs; i++) {
            int sl = (int)std::round(std::exp(seq_log_dist_(rng_)));
            sl = std::max(1, std::min(MAX_SEQ_LEN, sl));
            seq_lens[i] = sl;
            total += sl;
        }
        req.total_tokens = total;

        // input_ids: all zeros (token content doesn't affect perf)
        req.input_ids.resize(total, 0);

        // actual_seq_lengths: cumulative
        req.actual_seq_lengths.resize(bs);
        int acc = 0;
        for (int i = 0; i < bs; i++) {
            acc += seq_lens[i];
            req.actual_seq_lengths[i] = acc;
        }

        // position_ids: each seq restarts from 0
        std::vector<int64_t> pos_ids(total);
        int idx = 0;
        for (int i = 0; i < bs; i++) {
            for (int p = 0; p < seq_lens[i]; p++) {
                pos_ids[idx++] = p;
            }
        }

        // gather cos/sin
        table_.gather(pos_ids, req.cos_data, req.sin_data);

        return req;
    }

private:
    std::mt19937 rng_;
    const CosSinTable& table_;
    std::normal_distribution<double> batch_dist_;
    std::normal_distribution<double> seq_log_dist_;  // log-space normal
};

// ============================================================================
// Thread-safe queue
// ============================================================================

class BlockingQueue {
public:
    BlockingQueue(size_t max_size) : max_size_(max_size), closed_(false) {}

    void push(Request* req) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_full_.wait(lock, [this] { return q_.size() < max_size_ || closed_; });
        if (closed_) return;
        q_.push(req);
        cv_not_empty_.notify_one();
    }

    Request* pop(int timeout_ms = 500) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (cv_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                [this] { return !q_.empty() || closed_; })) {
            if (q_.empty()) return nullptr;
            Request* req = q_.front();
            q_.pop();
            cv_not_full_.notify_one();
            return req;
        }
        return nullptr;  // timeout
    }

    void close() {
        std::lock_guard<std::mutex> lock(mtx_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.size();
    }

private:
    std::queue<Request*> q_;
    mutable std::mutex mtx_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    size_t max_size_;
    bool closed_;
};

// ============================================================================
// Stats
// ============================================================================

struct LatencyStats {
    std::vector<double> e2e_times;          // enqueue → infer_end
    std::vector<double> e2e_prep_times;     // arrive → infer_end (incl. prep)
    std::vector<double> prep_times;         // arrive → enqueue (gather cos/sin etc.)
    std::vector<double> queue_times;        // enqueue → dequeue
    std::vector<double> infer_times;        // infer_start → infer_end
    long total_tokens = 0;
    std::map<int, std::vector<double>> per_stream;
    std::mutex mtx;

    void record(const Request& req, int stream_id) {
        std::lock_guard<std::mutex> lock(mtx);
        e2e_times.push_back(elapsed_ms(req.enqueue_time, req.infer_end));
        e2e_prep_times.push_back(elapsed_ms(req.arrive_time, req.infer_end));
        prep_times.push_back(elapsed_ms(req.arrive_time, req.enqueue_time));
        queue_times.push_back(elapsed_ms(req.enqueue_time, req.dequeue_time));
        infer_times.push_back(elapsed_ms(req.infer_start, req.infer_end));
        total_tokens += req.total_tokens;
        per_stream[stream_id].push_back(elapsed_ms(req.infer_start, req.infer_end));
    }

    static double percentile(std::vector<double> data, double p) {
        if (data.empty()) return 0.0;
        std::sort(data.begin(), data.end());
        size_t idx = (size_t)(data.size() * p / 100.0);
        if (idx >= data.size()) idx = data.size() - 1;
        return data[idx];
    }

    static double mean(const std::vector<double>& data) {
        if (data.empty()) return 0.0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    void report(double total_ms, int num_streams, int warmup) const {
        int n = e2e_times.size();
        if (n == 0) {
            std::cout << "[WARN] No requests completed" << std::endl;
            return;
        }
        double qps = n / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);

        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << "Throughput Benchmark Results (C++)\n";
        std::cout << std::string(72, '=') << "\n";
        std::cout << "  Streams:          " << num_streams << "\n";
        std::cout << "  Requests:         " << n << " (warmup=" << warmup << ")\n";
        std::cout << "  Total Time:       " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        std::cout << "  Total Tokens:     " << total_tokens << "\n";
        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  QPS:              " << std::fixed << std::setprecision(2) << qps << " req/s\n";
        std::cout << "  Token Throughput: " << std::fixed << std::setprecision(0) << tps << " tokens/s\n";
        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  Latency (ms):\n";

        auto print_row = [](const char* name, const std::vector<double>& data) {
            std::cout << "  " << std::left << std::setw(20) << name
                      << " avg=" << std::fixed << std::setprecision(2) << std::setw(8) << mean(data)
                      << " p50=" << std::setw(8) << percentile(data, 50)
                      << " p90=" << std::setw(8) << percentile(data, 90)
                      << " p99=" << std::setw(8) << percentile(data, 99)
                      << " max=" << std::setw(8) << (data.empty() ? 0.0 : *std::max_element(data.begin(), data.end()))
                      << "\n";
        };
        print_row("E2E (incl prep)", e2e_prep_times);
        print_row("  Prep", prep_times);
        print_row("  Queue Wait", queue_times);
        print_row("  Inference", infer_times);

        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  Per-Stream:\n";
        for (auto& kv : per_stream) {
            std::cout << "    Stream " << kv.first << ": " << kv.second.size()
                      << " reqs, avg_infer=" << std::fixed << std::setprecision(2)
                      << mean(kv.second) << "ms\n";
        }
        std::cout << std::string(72, '=') << "\n\n";
    }

    struct Summary {
        double qps;
        double tps;
        double e2e_avg, e2e_p99;
        double e2e_prep_avg, e2e_prep_p99;
        double prep_avg;
        double infer_avg, infer_p99;
    };

    Summary get_summary(double total_ms) const {
        double qps = e2e_times.size() / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);
        return {qps, tps,
                mean(e2e_times), percentile(e2e_times, 99),
                mean(e2e_prep_times), percentile(e2e_prep_times, 99),
                mean(prep_times),
                mean(infer_times), percentile(infer_times, 99)};
    }
};

// ============================================================================
// StreamContext: per-stream ACL resources
// ============================================================================

class StreamContext {
public:
    int stream_id;
    uint32_t model_id;  // each stream has its own model instance
    aclmdlDesc* model_desc;
    aclrtStream stream;
    aclmdlDataset* input_dataset;
    aclmdlDataset* output_dataset;
    std::vector<void*> input_buffers;
    std::vector<size_t> input_max_sizes;
    std::vector<void*> output_buffers;
    std::vector<size_t> output_max_sizes;

    // Input specs: (dtype, max_shape, max_bytes)
    struct InputSpec {
        aclDataType dtype;
        std::vector<int64_t> max_shape;
        size_t max_bytes;
    };

    std::vector<InputSpec> input_specs;
    size_t max_output_bytes;

    StreamContext(int sid, const std::string& model_path, int device_id)
        : stream_id(sid), model_id(0), model_desc(nullptr),
          stream(nullptr), input_dataset(nullptr), output_dataset(nullptr),
          max_output_bytes(0) {
        // 4 inputs matching OM model
        input_specs = {
            {ACL_INT64,   {MAX_BATCH_SIZE},                    (size_t)MAX_BATCH_SIZE * 8},
            {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM},     (size_t)1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2},
            {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM},     (size_t)1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2},
            {ACL_INT64,   {MAX_TOTAL_TOKENS},                  (size_t)MAX_TOTAL_TOKENS * 8},
        };
        max_output_bytes = (size_t)MAX_BATCH_SIZE * VOCAB_SIZE * 2;  // [N, vocab] fp16
        model_path_ = model_path;
        device_id_ = device_id;
    }

    int init() {
        // Load own model instance
        aclError ret = aclmdlLoadFromFile(model_path_.c_str(), &model_id);
        ACL_CHECK(ret, "load_model stream " << stream_id);
        model_desc = aclmdlCreateDesc();
        ret = aclmdlGetDesc(model_desc, model_id);
        ACL_CHECK(ret, "get_desc stream " << stream_id);

        ret = aclrtCreateStream(&stream);
        ACL_CHECK(ret, "create_stream " << stream_id);

        input_dataset = aclmdlCreateDataset();
        output_dataset = aclmdlCreateDataset();
        if (!input_dataset || !output_dataset) return ACL_ERROR_INTERNAL_ERROR;

        for (auto& spec : input_specs) {
            void* ptr = nullptr;
            ret = aclrtMalloc(&ptr, spec.max_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            ACL_CHECK(ret, "malloc input stream " << stream_id);
            aclDataBuffer* buf = aclCreateDataBuffer(ptr, spec.max_bytes);
            if (!buf) { aclrtFree(ptr); return ACL_ERROR_INTERNAL_ERROR; }
            ret = aclmdlAddDatasetBuffer(input_dataset, buf);
            ACL_CHECK(ret, "add input buffer");
            input_buffers.push_back(ptr);
            input_max_sizes.push_back(spec.max_bytes);
        }

        void* out_ptr = nullptr;
        ret = aclrtMalloc(&out_ptr, max_output_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        ACL_CHECK(ret, "malloc output stream " << stream_id);
        aclDataBuffer* out_buf = aclCreateDataBuffer(out_ptr, max_output_bytes);
        if (!out_buf) { aclrtFree(out_ptr); return ACL_ERROR_INTERNAL_ERROR; }
        ret = aclmdlAddDatasetBuffer(output_dataset, out_buf);
        ACL_CHECK(ret, "add output buffer");
        output_buffers.push_back(out_ptr);
        output_max_sizes.push_back(max_output_bytes);

        return ACL_SUCCESS;
    }

    // Copy request data to device buffers and set dynamic tensor descs
    int set_inputs(const Request& req) {
        struct InputData {
            const void* ptr;
            size_t bytes;
            aclDataType dtype;
            std::vector<int64_t> shape;
        };

        InputData inputs[4] = {
            {req.actual_seq_lengths.data(), req.actual_seq_lengths.size() * 8, ACL_INT64,
             {static_cast<int64_t>(req.actual_seq_lengths.size())}},
            {req.cos_data.data(), req.cos_data.size() * 2, ACL_FLOAT16,
             {1, static_cast<int64_t>(req.total_tokens), HEAD_DIM}},
            {req.sin_data.data(), req.sin_data.size() * 2, ACL_FLOAT16,
             {1, static_cast<int64_t>(req.total_tokens), HEAD_DIM}},
            {req.input_ids.data(), req.input_ids.size() * 8, ACL_INT64,
             {static_cast<int64_t>(req.total_tokens)}},
        };

        for (int i = 0; i < 4; i++) {
            if (inputs[i].bytes > input_max_sizes[i]) {
                std::cerr << "[ERROR] Input " << i << " size " << inputs[i].bytes
                          << " > buffer " << input_max_sizes[i] << std::endl;
                return ACL_ERROR_INVALID_PARAM;
            }
            aclError ret = aclrtMemcpy(input_buffers[i], input_max_sizes[i],
                                       inputs[i].ptr, inputs[i].bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE);
            ACL_CHECK(ret, "H2D input[" << i << "]");

            aclTensorDesc* desc = aclCreateTensorDesc(
                inputs[i].dtype,
                static_cast<int32_t>(inputs[i].shape.size()),
                inputs[i].shape.data(),
                ACL_FORMAT_ND);
            if (!desc) return ACL_ERROR_INTERNAL_ERROR;
            ret = aclmdlSetDatasetTensorDesc(input_dataset, desc, i);
            aclDestroyTensorDesc(desc);
            ACL_CHECK(ret, "set_desc input[" << i << "]");
        }
        return ACL_SUCCESS;
    }

    int execute() {
        aclError ret = aclmdlExecuteAsync(model_id, input_dataset, output_dataset, stream);
        ACL_CHECK(ret, "execute_async stream " << stream_id);
        ret = aclrtSynchronizeStream(stream);
        ACL_CHECK(ret, "sync_stream " << stream_id);
        return ACL_SUCCESS;
    }

    void cleanup() {
        auto free_dataset = [](aclmdlDataset* ds) {
            if (!ds) return;
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(ds); i++) {
                aclDataBuffer* buf = aclmdlGetDatasetBuffer(ds, i);
                if (buf) {
                    void* ptr = aclGetDataBufferAddr(buf);
                    if (ptr) aclrtFree(ptr);
                    aclDestroyDataBuffer(buf);
                }
            }
            aclmdlDestroyDataset(ds);
        };
        free_dataset(input_dataset);
        input_dataset = nullptr;
        free_dataset(output_dataset);
        output_dataset = nullptr;
        if (stream) {
            aclrtSynchronizeStream(stream);
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (model_desc) {
            aclmdlDestroyDesc(model_desc);
            model_desc = nullptr;
        }
        if (model_id != 0) {
            aclmdlUnload(model_id);
            model_id = 0;
        }
    }

private:
    std::string model_path_;
    int device_id_;
};

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchResult {
    int num_streams;
    double total_ms;
    LatencyStats::Summary summary;
};

static int run_benchmark(const std::string& model_path,
                         int num_streams, int total_requests, int warmup,
                         const CosSinTable& cos_sin_table, int device_id,
                         aclrtContext acl_ctx,
                         BenchResult& result) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Config: streams=" << num_streams << ", requests=" << total_requests
              << ", warmup=" << warmup << ", device=" << device_id << "\n";
    std::cout << std::string(72, '=') << "\n";

    // Set context for this thread
    aclError ret = aclrtSetCurrentContext(acl_ctx);
    ACL_CHECK(ret, "set_current_context");

    // Create stream contexts (each loads its own model instance)
    std::vector<std::unique_ptr<StreamContext>> streams;
    for (int s = 0; s < num_streams; s++) {
        auto ctx = std::make_unique<StreamContext>(s, model_path, device_id);
        int ret = ctx->init();
        if (ret != ACL_SUCCESS) {
            std::cerr << "[ERROR] Failed to init stream " << s << std::endl;
            return ret;
        }
        streams.push_back(std::move(ctx));
    }
    std::cout << "[INFO] Created " << num_streams << " stream contexts (each with own model instance)" << std::endl;

    // Warmup
    if (warmup > 0) {
        std::cout << "[INFO] Warmup (" << warmup << " requests)..." << std::endl;
        RequestGenerator gen(0, cos_sin_table);
        for (int i = 0; i < warmup; i++) {
            Request req = gen.generate(i, Clock::now());
            int idx = i % num_streams;
            int ret = streams[idx]->set_inputs(req);
            if (ret != ACL_SUCCESS) return ret;
            ret = streams[idx]->execute();
            if (ret != ACL_SUCCESS) return ret;
        }
        std::cout << "[INFO] Warmup done" << std::endl;
    }

    int queue_depth = num_streams * 4;
    BlockingQueue queue(queue_depth);
    LatencyStats stats;
    std::atomic<bool> running(true);
    std::atomic<int> completed(0);
    std::atomic<int> errors(0);
    std::vector<std::thread> consumers;

    // Consumer threads
    for (int s = 0; s < num_streams; s++) {
        consumers.emplace_back([&, s]() {
            aclrtSetCurrentContext(acl_ctx);
            StreamContext* ctx = streams[s].get();
            while (true) {
                Request* req_ptr = queue.pop(500);
                if (req_ptr == nullptr) {
                    if (!running.load()) break;
                    continue;
                }
                req_ptr->dequeue_time = Clock::now();
                int ret = ctx->set_inputs(*req_ptr);
                if (ret != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Stream " << s << " set_inputs failed" << std::endl;
                    delete req_ptr;
                    errors.fetch_add(1);
                    break;
                }
                req_ptr->infer_start = Clock::now();
                ret = ctx->execute();
                req_ptr->infer_end = Clock::now();
                if (ret != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Stream " << s << " execute failed" << std::endl;
                    delete req_ptr;
                    errors.fetch_add(1);
                    break;
                }
                stats.record(*req_ptr, s);
                completed.fetch_add(1);
                delete req_ptr;
            }
        });
    }

    // Producer thread + timing
    TimePoint bench_start = Clock::now();
    std::thread producer([&]() {
        aclrtSetCurrentContext(acl_ctx);
        RequestGenerator gen(42, cos_sin_table);
        for (int i = 0; i < total_requests; i++) {
            TimePoint arrive = Clock::now();
            Request* req = new Request(gen.generate(i, arrive));  // prep: gather cos/sin, build inputs
            req->enqueue_time = Clock::now();
            queue.push(req);
        }
    });

    producer.join();

    // Wait for all requests to complete
    while (completed.load() + errors.load() < total_requests) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    TimePoint bench_end = Clock::now();

    // Stop consumers
    running.store(false);
    queue.close();
    for (auto& t : consumers) {
        t.join();
    }

    double total_ms = elapsed_ms(bench_start, bench_end);
    stats.report(total_ms, num_streams, warmup);

    if (errors.load() > 0) {
        std::cout << "[WARN] " << errors.load() << " errors occurred" << std::endl;
    }

    // Cleanup
    for (auto& ctx : streams) {
        ctx->cleanup();
    }

    result.num_streams = num_streams;
    result.total_ms = total_ms;
    result.summary = stats.get_summary(total_ms);

    return ACL_SUCCESS;
}

// ============================================================================
// Sweep table
// ============================================================================

static void print_sweep_table(const std::vector<BenchResult>& results) {
    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << "Sweep Results: QPS vs Stream Count\n";
    std::cout << std::string(110, '=') << "\n";
    std::cout << std::right
              << std::setw(7) << "Streams" << " | "
              << std::setw(10) << "QPS" << " | "
              << std::setw(12) << "Token/s" << " | "
              << std::setw(10) << "Prep avg" << " | "
              << std::setw(12) << "E2E avg" << " | "
              << std::setw(12) << "E2E+prep p99" << " | "
              << std::setw(10) << "Infer avg" << " | "
              << std::setw(10) << "Infer p99" << "\n";
    std::cout << std::string(110, '-') << "\n";
    for (auto& r : results) {
        std::cout << std::right
                  << std::setw(7) << r.num_streams << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.summary.qps << " | "
                  << std::setw(12) << std::setprecision(0) << r.summary.tps << " | "
                  << std::setw(10) << std::setprecision(3) << r.summary.prep_avg << " | "
                  << std::setw(12) << std::setprecision(2) << r.summary.e2e_prep_avg << " | "
                  << std::setw(12) << r.summary.e2e_prep_p99 << " | "
                  << std::setw(10) << r.summary.infer_avg << " | "
                  << std::setw(10) << r.summary.infer_p99 << "\n";
    }
    std::cout << std::string(110, '=') << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nMulti-stream throughput benchmark for Qwen2.5-0.5B OM model.\n"
              << "\nOptions:\n"
              << "  --model <path>       Path to .om model (required)\n"
              << "  --streams <N>        Number of streams (default: 4)\n"
              << "  --sweep <list>       Comma-separated stream counts (e.g. 1,2,4,8)\n"
              << "  --requests <N>       Total requests (default: 10000)\n"
              << "  --device-id <id>     NPU device ID (default: 0)\n"
              << "  --warmup <N>         Warmup requests (default: 50)\n"
              << "  --queue-depth <N>    Queue depth (default: streams * 4)\n"
              << "  -h, --help           Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " --model model.om --streams 4 --requests 10000 --device-id 2\n"
              << "  " << prog << " --model model.om --sweep 1,2,4,8,16 --requests 10000 --device-id 2\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    int num_streams = 4;
    std::string sweep;
    int total_requests = 10000;
    int device_id = 0;
    int warmup = 50;
    int queue_depth = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--streams" && i + 1 < argc) {
            num_streams = std::stoi(argv[++i]);
        } else if (arg == "--sweep" && i + 1 < argc) {
            sweep = argv[++i];
        } else if (arg == "--requests" && i + 1 < argc) {
            total_requests = std::stoi(argv[++i]);
        } else if (arg == "--device-id" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--queue-depth" && i + 1 < argc) {
            queue_depth = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "[ERROR] --model is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::vector<int> stream_counts;
    if (!sweep.empty()) {
        std::stringstream ss(sweep);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            stream_counts.push_back(std::stoi(tok));
        }
    } else {
        stream_counts.push_back(num_streams);
    }

    // Init ACL
    aclError ret = aclInit(nullptr);
    ACL_CHECK(ret, "aclInit");
    ret = aclrtSetDevice(device_id);
    ACL_CHECK(ret, "set_device");

    // Create explicit context for multi-threading
    aclrtContext acl_ctx;
    ret = aclrtCreateContext(&acl_ctx, device_id);
    ACL_CHECK(ret, "create_context");

    // Precompute cos/sin table
    CosSinTable cos_sin_table;
    cos_sin_table.init(MAX_SEQ_LEN);
    std::cout << "[INFO] Cos/sin table: [" << MAX_SEQ_LEN << ", " << HEAD_DIM << "]" << std::endl;

    // Run benchmarks
    std::vector<BenchResult> results;
    for (int n : stream_counts) {
        BenchResult r;
        ret = run_benchmark(model_path, n, total_requests, warmup,
                               cos_sin_table, device_id, acl_ctx, r);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[ERROR] Benchmark failed for streams=" << n << std::endl;
            break;
        }
        results.push_back(r);
    }

    if (results.size() > 1) {
        print_sweep_table(results);
    }

    // Cleanup
    aclrtDestroyContext(acl_ctx);
    aclrtResetDevice(device_id);
    aclFinalize();

    return 0;
}
