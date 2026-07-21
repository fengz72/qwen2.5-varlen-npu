#include <iostream>
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
static const int    MAX_BATCH_SIZE     = 11;
static const int    MAX_SEQ_LEN        = 218;
static const int    MAX_TOTAL_TOKENS   = MAX_BATCH_SIZE * MAX_SEQ_LEN;

static const double BATCH_AVG   = 9.8;
static const double BATCH_STD   = 0.35;
static const bool   BATCH_FIXED = true;
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
    std::vector<float> inv_freq;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    int max_len;
    int dim;

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
                cos_out[i * dim + d] = float_to_fp16(c);
                sin_out[i * dim + d] = float_to_fp16(s);
            }
        }
    }
};

// ============================================================================
// Request with per-stage timing
// ============================================================================

struct Request {
    int req_id;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> actual_seq_lengths;
    std::vector<uint16_t> cos_data;
    std::vector<uint16_t> sin_data;
    int total_tokens;
    int batch_size;

    // Per-stage timing
    TimePoint arrive;         // iteration start
    TimePoint gen_done;       // after data generation
    TimePoint h2d_done;       // after H2D + set dynamic shape
    TimePoint execute_done;   // after execute (async + sync)
    TimePoint d2h_done;       // after D2H (E2E end)
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
        req.arrive = arrive_time;

        int bs;
        if (BATCH_FIXED) {
            bs = BATCH_FIXED_VAL;
        } else {
            bs = (int)std::round(batch_dist_(rng_));
            bs = std::max(1, std::min(MAX_BATCH_SIZE, bs));
        }
        req.batch_size = bs;

        std::vector<int> seq_lens(bs);
        int total = 0;
        for (int i = 0; i < bs; i++) {
            int sl = (int)std::round(std::exp(seq_log_dist_(rng_)));
            sl = std::max(1, std::min(MAX_SEQ_LEN, sl));
            seq_lens[i] = sl;
            total += sl;
        }
        req.total_tokens = total;

        req.input_ids.resize(total, 0);

        req.actual_seq_lengths.resize(bs);
        int acc = 0;
        for (int i = 0; i < bs; i++) {
            acc += seq_lens[i];
            req.actual_seq_lengths[i] = acc;
        }

        std::vector<int64_t> pos_ids(total);
        int idx = 0;
        for (int i = 0; i < bs; i++) {
            for (int p = 0; p < seq_lens[i]; p++) {
                pos_ids[idx++] = p;
            }
        }

        table_.gather(pos_ids, req.cos_data, req.sin_data);

        return req;
    }

private:
    std::mt19937 rng_;
    const CosSinTable& table_;
    std::normal_distribution<double> batch_dist_;
    std::normal_distribution<double> seq_log_dist_;
};

// ============================================================================
// Latency stats (per-stage breakdown)
// ============================================================================

struct LatencyStats {
    std::vector<double> gen_times;       // arrive → gen_done
    std::vector<double> h2d_times;       // gen_done → h2d_done
    std::vector<double> exec_times;      // h2d_done → execute_done
    std::vector<double> d2h_times;       // execute_done → d2h_done
    std::vector<double> e2e_times;       // arrive → d2h_done
    long total_tokens = 0;
    std::map<int, std::vector<double>> per_thread;
    std::mutex mtx;

    void record(const Request& req, int thread_id) {
        std::lock_guard<std::mutex> lock(mtx);
        gen_times.push_back(elapsed_ms(req.arrive, req.gen_done));
        h2d_times.push_back(elapsed_ms(req.gen_done, req.h2d_done));
        exec_times.push_back(elapsed_ms(req.h2d_done, req.execute_done));
        d2h_times.push_back(elapsed_ms(req.execute_done, req.d2h_done));
        e2e_times.push_back(elapsed_ms(req.arrive, req.d2h_done));
        total_tokens += req.total_tokens;
        per_thread[thread_id].push_back(elapsed_ms(req.arrive, req.d2h_done));
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

    void report(double total_ms, int num_threads, int warmup) const {
        int n = e2e_times.size();
        if (n == 0) {
            std::cout << "[WARN] No requests completed" << std::endl;
            return;
        }
        double qps = n / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);

        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << "Latency Benchmark Results (Independent Threads, C++)\n";
        std::cout << std::string(72, '=') << "\n";
        std::cout << "  Threads:          " << num_threads << "\n";
        std::cout << "  Requests:         " << n << " (warmup=" << warmup << ")\n";
        std::cout << "  Total Time:       " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        std::cout << "  Total Tokens:     " << total_tokens << "\n";
        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  QPS:              " << std::fixed << std::setprecision(2) << qps << " req/s\n";
        std::cout << "  Token Throughput: " << std::fixed << std::setprecision(0) << tps << " tokens/s\n";
        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  Latency Breakdown (ms):\n";

        auto print_row = [](const char* name, const std::vector<double>& data) {
            std::cout << "  " << std::left << std::setw(20) << name
                      << " avg=" << std::fixed << std::setprecision(3) << std::setw(8) << mean(data)
                      << " p50=" << std::setw(8) << percentile(data, 50)
                      << " p90=" << std::setw(8) << percentile(data, 90)
                      << " p99=" << std::setw(8) << percentile(data, 99)
                      << " max=" << std::setw(8) << (data.empty() ? 0.0 : *std::max_element(data.begin(), data.end()))
                      << "\n";
        };
        print_row("E2E (full)", e2e_times);
        print_row("  Data Gen", gen_times);
        print_row("  H2D", h2d_times);
        print_row("  Execute", exec_times);
        print_row("  D2H", d2h_times);

        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  Per-Thread E2E:\n";
        for (auto& kv : per_thread) {
            std::cout << "    Thread " << kv.first << ": " << kv.second.size()
                      << " reqs, avg_e2e=" << std::fixed << std::setprecision(3)
                      << mean(kv.second) << "ms\n";
        }
        std::cout << std::string(72, '=') << "\n\n";
    }

    struct Summary {
        double qps;
        double tps;
        double e2e_avg, e2e_p99;
        double gen_avg;
        double h2d_avg;
        double exec_avg, exec_p99;
        double d2h_avg;
    };

    Summary get_summary(double total_ms) const {
        double qps = e2e_times.size() / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);
        return {qps, tps,
                mean(e2e_times), percentile(e2e_times, 99),
                mean(gen_times),
                mean(h2d_times),
                mean(exec_times), percentile(exec_times, 99),
                mean(d2h_times)};
    }
};

// ============================================================================
// StreamContext: per-thread ACL resources (with D2H support)
// ============================================================================

class StreamContext {
public:
    int thread_id;
    uint32_t model_id;
    aclmdlDesc* model_desc;
    aclrtStream stream;
    aclmdlDataset* input_dataset;
    aclmdlDataset* output_dataset;
    std::vector<void*> input_buffers;
    std::vector<size_t> input_max_sizes;
    void* output_buffer;
    size_t output_max_bytes;

    // Host buffer for D2H
    std::vector<uint8_t> host_output;

    struct InputSpec {
        aclDataType dtype;
        std::vector<int64_t> max_shape;
        size_t max_bytes;
    };

    std::vector<InputSpec> input_specs;

    StreamContext(int tid, const std::string& model_path, int device_id)
        : thread_id(tid), model_id(0), model_desc(nullptr),
          stream(nullptr), input_dataset(nullptr), output_dataset(nullptr),
          output_buffer(nullptr), output_max_bytes(0) {
        input_specs = {
            {ACL_INT64,   {MAX_BATCH_SIZE},                    (size_t)MAX_BATCH_SIZE * 8},
            {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM},     (size_t)1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2},
            {ACL_FLOAT16, {1, MAX_TOTAL_TOKENS, HEAD_DIM},     (size_t)1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2},
            {ACL_INT64,   {MAX_TOTAL_TOKENS},                  (size_t)MAX_TOTAL_TOKENS * 8},
        };
        output_max_bytes = (size_t)MAX_BATCH_SIZE * VOCAB_SIZE * 2;
        model_path_ = model_path;
        device_id_ = device_id;
        host_output.resize(output_max_bytes);
    }

    int init() {
        aclError ret = aclmdlLoadFromFile(model_path_.c_str(), &model_id);
        ACL_CHECK(ret, "load_model thread " << thread_id);
        model_desc = aclmdlCreateDesc();
        ret = aclmdlGetDesc(model_desc, model_id);
        ACL_CHECK(ret, "get_desc thread " << thread_id);

        ret = aclrtCreateStream(&stream);
        ACL_CHECK(ret, "create_stream " << thread_id);

        input_dataset = aclmdlCreateDataset();
        output_dataset = aclmdlCreateDataset();
        if (!input_dataset || !output_dataset) return ACL_ERROR_INTERNAL_ERROR;

        for (auto& spec : input_specs) {
            void* ptr = nullptr;
            ret = aclrtMalloc(&ptr, spec.max_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            ACL_CHECK(ret, "malloc input thread " << thread_id);
            aclDataBuffer* buf = aclCreateDataBuffer(ptr, spec.max_bytes);
            if (!buf) { aclrtFree(ptr); return ACL_ERROR_INTERNAL_ERROR; }
            ret = aclmdlAddDatasetBuffer(input_dataset, buf);
            ACL_CHECK(ret, "add input buffer");
            input_buffers.push_back(ptr);
            input_max_sizes.push_back(spec.max_bytes);
        }

        ret = aclrtMalloc(&output_buffer, output_max_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        ACL_CHECK(ret, "malloc output thread " << thread_id);
        aclDataBuffer* out_buf = aclCreateDataBuffer(output_buffer, output_max_bytes);
        if (!out_buf) { aclrtFree(output_buffer); return ACL_ERROR_INTERNAL_ERROR; }
        ret = aclmdlAddDatasetBuffer(output_dataset, out_buf);
        ACL_CHECK(ret, "add output buffer");

        return ACL_SUCCESS;
    }

    // H2D: copy request data to device + set dynamic tensor descs
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
        ACL_CHECK(ret, "execute_async thread " << thread_id);
        ret = aclrtSynchronizeStream(stream);
        ACL_CHECK(ret, "sync_stream " << thread_id);
        return ACL_SUCCESS;
    }

    // D2H: copy output [N, vocab] fp16 from device to host
    int d2h(int batch_size) {
        size_t out_bytes = (size_t)batch_size * VOCAB_SIZE * 2;
        if (out_bytes > output_max_bytes) out_bytes = output_max_bytes;
        aclError ret = aclrtMemcpy(host_output.data(), host_output.size(),
                                   output_buffer, out_bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        ACL_CHECK(ret, "D2H output thread " << thread_id);
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
// Benchmark runner (independent threads, no queue)
// ============================================================================

struct BenchResult {
    int num_threads;
    double total_ms;
    LatencyStats::Summary summary;
};

static int run_benchmark(const std::string& model_path,
                         int num_threads, int total_requests, int warmup,
                         const CosSinTable& cos_sin_table, int device_id,
                         aclrtContext acl_ctx,
                         BenchResult& result) {
    int requests_per_thread = (total_requests + num_threads - 1) / num_threads;

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "Config: threads=" << num_threads
              << ", total_requests=" << total_requests
              << ", reqs/thread=" << requests_per_thread
              << ", warmup=" << warmup
              << ", device=" << device_id << "\n";
    std::cout << std::string(72, '=') << "\n";

    aclError ret = aclrtSetCurrentContext(acl_ctx);
    ACL_CHECK(ret, "set_current_context");

    // Create stream contexts (each loads its own model instance)
    std::vector<std::unique_ptr<StreamContext>> threads_ctx;
    for (int t = 0; t < num_threads; t++) {
        auto ctx = std::make_unique<StreamContext>(t, model_path, device_id);
        int r = ctx->init();
        if (r != ACL_SUCCESS) {
            std::cerr << "[ERROR] Failed to init thread ctx " << t << std::endl;
            return r;
        }
        threads_ctx.push_back(std::move(ctx));
    }
    std::cout << "[INFO] Created " << num_threads << " thread contexts (each with own model instance)" << std::endl;

    // Warmup (single thread, sequential)
    if (warmup > 0) {
        std::cout << "[INFO] Warmup (" << warmup << " requests)..." << std::endl;
        RequestGenerator gen(0, cos_sin_table);
        for (int i = 0; i < warmup; i++) {
            Request req = gen.generate(i, Clock::now());
            int r = threads_ctx[0]->set_inputs(req);
            if (r != ACL_SUCCESS) return r;
            r = threads_ctx[0]->execute();
            if (r != ACL_SUCCESS) return r;
            r = threads_ctx[0]->d2h(req.batch_size);
            if (r != ACL_SUCCESS) return r;
        }
        std::cout << "[INFO] Warmup done" << std::endl;
    }

    LatencyStats stats;
    std::atomic<int> errors(0);

    // Launch N independent threads
    TimePoint bench_start = Clock::now();

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t]() {
            aclrtSetCurrentContext(acl_ctx);
            StreamContext* ctx = threads_ctx[t].get();
            RequestGenerator gen(42 + t * 1000, cos_sin_table);

            for (int i = 0; i < requests_per_thread; i++) {
                int req_id = t * requests_per_thread + i;
                Request req = gen.generate(req_id, Clock::now());

                req.gen_done = Clock::now();

                int r = ctx->set_inputs(req);
                if (r != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Thread " << t << " set_inputs failed at req " << i << std::endl;
                    errors.fetch_add(1);
                    return;
                }
                req.h2d_done = Clock::now();

                r = ctx->execute();
                if (r != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Thread " << t << " execute failed at req " << i << std::endl;
                    errors.fetch_add(1);
                    return;
                }
                req.execute_done = Clock::now();

                r = ctx->d2h(req.batch_size);
                if (r != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Thread " << t << " d2h failed at req " << i << std::endl;
                    errors.fetch_add(1);
                    return;
                }
                req.d2h_done = Clock::now();

                stats.record(req, t);
            }
        });
    }

    // Wait for all threads
    for (auto& w : workers) {
        w.join();
    }
    TimePoint bench_end = Clock::now();

    double total_ms = elapsed_ms(bench_start, bench_end);
    stats.report(total_ms, num_threads, warmup);

    if (errors.load() > 0) {
        std::cout << "[WARN] " << errors.load() << " errors occurred" << std::endl;
    }

    // Cleanup
    for (auto& ctx : threads_ctx) {
        ctx->cleanup();
    }

    result.num_threads = num_threads;
    result.total_ms = total_ms;
    result.summary = stats.get_summary(total_ms);

    return ACL_SUCCESS;
}

// ============================================================================
// Sweep table
// ============================================================================

static void print_sweep_table(const std::vector<BenchResult>& results) {
    std::cout << "\n" << std::string(130, '=') << "\n";
    std::cout << "Sweep Results: Latency & Throughput vs Thread Count (Independent Threads)\n";
    std::cout << std::string(130, '=') << "\n";
    std::cout << std::right
              << std::setw(7) << "Threads" << " | "
              << std::setw(10) << "QPS" << " | "
              << std::setw(12) << "Token/s" << " | "
              << std::setw(10) << "Gen avg" << " | "
              << std::setw(10) << "H2D avg" << " | "
              << std::setw(10) << "Exec avg" << " | "
              << std::setw(10) << "Exec p99" << " | "
              << std::setw(10) << "D2H avg" << " | "
              << std::setw(10) << "E2E avg" << " | "
              << std::setw(10) << "E2E p99" << "\n";
    std::cout << std::string(130, '-') << "\n";
    for (auto& r : results) {
        std::cout << std::right
                  << std::setw(7) << r.num_threads << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.summary.qps << " | "
                  << std::setw(12) << std::setprecision(0) << r.summary.tps << " | "
                  << std::setw(10) << std::setprecision(3) << r.summary.gen_avg << " | "
                  << std::setw(10) << r.summary.h2d_avg << " | "
                  << std::setw(10) << std::setprecision(2) << r.summary.exec_avg << " | "
                  << std::setw(10) << r.summary.exec_p99 << " | "
                  << std::setw(10) << std::setprecision(3) << r.summary.d2h_avg << " | "
                  << std::setw(10) << std::setprecision(2) << r.summary.e2e_avg << " | "
                  << std::setw(10) << r.summary.e2e_p99 << "\n";
    }
    std::cout << std::string(130, '=') << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nLatency & throughput benchmark (independent threads, no queue).\n"
              << "Each thread independently loops: data_gen → H2D → execute → D2H.\n"
              << "\nOptions:\n"
              << "  --model <path>       Path to .om model (required)\n"
              << "  --threads <N>        Number of independent threads (default: 4)\n"
              << "  --sweep <list>       Comma-separated thread counts (e.g. 1,2,4,8)\n"
              << "  --requests <N>       Total requests across all threads (default: 10000)\n"
              << "  --device-id <id>     NPU device ID (default: 0)\n"
              << "  --warmup <N>         Warmup requests (default: 50)\n"
              << "  -h, --help           Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " --model model.om --threads 4 --requests 10000 --device-id 2\n"
              << "  " << prog << " --model model.om --sweep 1,2,4,8,16 --requests 10000 --device-id 2\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    int num_threads = 4;
    std::string sweep;
    int total_requests = 10000;
    int device_id = 0;
    int warmup = 50;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--sweep" && i + 1 < argc) {
            sweep = argv[++i];
        } else if (arg == "--requests" && i + 1 < argc) {
            total_requests = std::stoi(argv[++i]);
        } else if (arg == "--device-id" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
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

    std::vector<int> thread_counts;
    if (!sweep.empty()) {
        std::stringstream ss(sweep);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            thread_counts.push_back(std::stoi(tok));
        }
    } else {
        thread_counts.push_back(num_threads);
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
    for (int n : thread_counts) {
        BenchResult r;
        ret = run_benchmark(model_path, n, total_requests, warmup,
                            cos_sin_table, device_id, acl_ctx, r);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[ERROR] Benchmark failed for threads=" << n << std::endl;
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
