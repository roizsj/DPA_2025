#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <utility>
#include <cmath>
using namespace std;

/* -----------------------------------------------------------
   Minimal IVF (Inverted File) with k-means (Lloyd + kmeans++)
   - Input format: Facebook/FAISS style fvecs/ivecs
     fvecs: per vector = int d; then d x float
     ivecs: per vector = int k; then k x int
   - Build: kmeans on learn set -> assign base -> inverted lists
   - Search: pick top-nprobe centroids by L2 to query, scan lists
   - Only standard C++ library, portable to aarch64/x86_64
   ----------------------------------------------------------- */

struct FVecs {
    int d = 0;
    size_t n = 0;
    vector<float> data; // flat: n * d
};
struct IVecs {
    int k = 0;
    size_t n = 0;
    vector<int> data; // flat: n * k
};

static bool write_raw_floats_to_block(const string& dev_path,
                                      uint64_t offset_bytes,
                                      const float* data,
                                      size_t nvec,
                                      int d)
{
    if (d <= 0 || nvec == 0) { cerr << "write: invalid n/d\n"; return false; }
    const size_t total_floats = nvec * (size_t)d;
    const size_t total_bytes  = total_floats * sizeof(float);

    std::fstream fio(dev_path, ios::in | ios::out | ios::binary);
    if (!fio) { cerr << "Failed to open for write: " << dev_path << "\n"; return false; }

    fio.seekp(static_cast<std::streamoff>(offset_bytes), ios::beg);
    if (!fio) { cerr << "seekp failed at offset " << offset_bytes << "\n"; return false; }

    fio.write(reinterpret_cast<const char*>(data), total_bytes);
    if (!fio) { cerr << "device write failed, wrote less than " << total_bytes << " bytes\n"; return false; }

    fio.flush();
    return true;
}

// 从块设备上批量读取 [start_idx, start_idx+count) 这些向量
// 布局假设为 headerless row-major：每条向量 d 个 float 连续存储
static bool read_vectors_from_device_batch(
    ifstream& fin,            // 已打开的设备文件（每个线程一个）
    uint64_t base_offset,     // 向量阵列的字节起始偏移
    size_t start_idx,         // 起始向量 id
    size_t count,             // 读取多少条
    int d,                    // 维度
    float* out                // 输出缓冲，大小 >= count*d
) {
    const uint64_t start_off = base_offset + (uint64_t)start_idx * (uint64_t)d * sizeof(float);
    fin.seekg(static_cast<std::streamoff>(start_off), ios::beg);
    if (!fin) return false;
    fin.read(reinterpret_cast<char*>(out), (std::streamsize)(count * (size_t)d * sizeof(float)));
    return (bool)fin;
}

// 把 centers (nlist*d floats) 写入设备
static bool write_centers_to_block(const string& dev_path,
                                   uint64_t offset_bytes,
                                   const vector<float>& centers) {
    if (centers.empty()) { cerr << "write centers: empty\n"; return false; }
    std::fstream fio(dev_path, ios::in | ios::out | ios::binary);
    if (!fio) { cerr << "Failed to open for write: " << dev_path << "\n"; return false; }
    fio.seekp(static_cast<std::streamoff>(offset_bytes), ios::beg);
    if (!fio) { cerr << "seekp failed at offset " << offset_bytes << "\n"; return false; }
    fio.write(reinterpret_cast<const char*>(centers.data()),
              (std::streamsize)(centers.size() * sizeof(float)));
    if (!fio) { cerr << "device write failed for centers\n"; return false; }
    fio.flush();
    return true;
}

// 从设备读 centers 到内存 (nlist*d floats)
static bool read_centers_from_block(const string& dev_path,
                                    uint64_t offset_bytes,
                                    int nlist, int d,
                                    vector<float>& centers_out) {
    if (nlist <= 0 || d <= 0) { cerr << "read centers: invalid nlist/d\n"; return false; }
    const size_t total = (size_t)nlist * (size_t)d;
    centers_out.resize(total);
    ifstream fin(dev_path, ios::binary);
    if (!fin) { cerr << "Failed to open for read: " << dev_path << "\n"; return false; }
    fin.seekg(static_cast<std::streamoff>(offset_bytes), ios::beg);
    if (!fin) { cerr << "seekg failed at offset " << offset_bytes << "\n"; return false; }
    fin.read(reinterpret_cast<char*>(centers_out.data()),
             (std::streamsize)(total * sizeof(float)));
    return (bool)fin;
}

static bool read_fvecs(const string& path, FVecs& out) {
    ifstream fin(path, ios::binary);
    if (!fin) return false;
    vector<float> all;
    int dim = -1;
    size_t n = 0;
    while (true) {
        int d;
        fin.read(reinterpret_cast<char*>(&d), 4);
        if (!fin) break; // eof
        if (dim == -1) dim = d;
        if (d != dim) { cerr << "Inconsistent d in " << path << "\n"; return false; }
        vector<float> buf(d);
        fin.read(reinterpret_cast<char*>(buf.data()), sizeof(float)*d);
        if (!fin) { cerr << "Truncated file: " << path << "\n"; return false; }
        all.insert(all.end(), buf.begin(), buf.end());
        ++n;
    }
    out.d = dim < 0 ? 0 : dim;
    out.n = n;
    out.data.swap(all);
    return true;
}

static bool read_ivecs(const string& path, IVecs& out) {
    ifstream fin(path, ios::binary);
    if (!fin) return false;
    vector<int> all;
    int k = -1;
    size_t n = 0;
    while (true) {
        int kk;
        fin.read(reinterpret_cast<char*>(&kk), 4);
        if (!fin) break;
        if (k == -1) k = kk;
        if (kk != k) { cerr << "Inconsistent k in " << path << "\n"; return false; }
        vector<int> buf(kk);
        fin.read(reinterpret_cast<char*>(buf.data()), sizeof(int)*kk);
        if (!fin) { cerr << "Truncated file: " << path << "\n"; return false; }
        all.insert(all.end(), buf.begin(), buf.end());
        ++n;
    }
    out.k = k < 0 ? 0 : k;
    out.n = n;
    out.data.swap(all);
    return true;
}

static bool load_fvecs_as_raw(const string& fpath, FVecs& out, int expect_d /* <=0则不校验 */) {
    if (!read_fvecs(fpath, out)) return false;
    if (expect_d > 0 && out.d != expect_d) {
        cerr << "Dim mismatch for " << fpath << ": got " << out.d << ", expect " << expect_d << "\n";
        return false;
    }
    return true;
}

inline float l2sq(const float* a, const float* b, int d) {
    // unrolled a bit for speed (still portable)
    float acc = 0.f;
    int i = 0;
    for (; i + 7 < d; i += 8) {
        float da0 = a[i+0] - b[i+0];
        float da1 = a[i+1] - b[i+1];
        float da2 = a[i+2] - b[i+2];
        float da3 = a[i+3] - b[i+3];
        float da4 = a[i+4] - b[i+4];
        float da5 = a[i+5] - b[i+5];
        float da6 = a[i+6] - b[i+6];
        float da7 = a[i+7] - b[i+7];
        acc += da0*da0 + da1*da1 + da2*da2 + da3*da3
             + da4*da4 + da5*da5 + da6*da6 + da7*da7;
    }
    for (; i < d; ++i) {
        float da = a[i] - b[i];
        acc += da*da;
    }
    return acc;
}

// kmeans++ init on learn set
static vector<float> kmeanspp_init(const FVecs& learn, int k, uint64_t seed=123) {
    int d = learn.d;
    size_t n = learn.n;
    const float* X = learn.data.data();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> uid(0, n-1);

    vector<float> centers(k * d, 0.f);
    // pick first center randomly
    size_t c0 = uid(rng);
    memcpy(centers.data(), X + c0*d, sizeof(float)*d);

    vector<double> mind2(n, numeric_limits<double>::infinity());

    for (int ci = 1; ci < k; ++ci) {
        // update distances to nearest already-chosen center
        for (size_t i = 0; i < n; ++i) {
            double dist = l2sq(X + i*d, centers.data() + (ci-1)*d, d);
            if (dist < mind2[i]) mind2[i] = dist;
        }
        // sample new center with prob ~ mind2
        long double sum = 0.0L;
        for (size_t i = 0; i < n; ++i) sum += mind2[i];
        long double r = (long double)std::uniform_real_distribution<double>(0.0,1.0)(rng) * sum;
        size_t chosen = n - 1;
        long double acc = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            acc += mind2[i];
            if (acc >= r) { chosen = i; break; }
        }
        memcpy(centers.data() + ci*d, X + chosen*d, sizeof(float)*d);
    }
    return centers;
}

struct KMeansResult {
    vector<float> centers; // nlist * d
    vector<int>   assign;  // learn.n assignments (optional, not used later)
};

static KMeansResult kmeans_train(const FVecs& learn, int nlist, int iters, int threads=1, uint64_t seed=123) {
    KMeansResult res;
    int d = learn.d;
    size_t n = learn.n;
    const float* X = learn.data.data();

    res.centers = kmeanspp_init(learn, nlist, seed);
    vector<int> assign(n, -1);
    vector<size_t> counts(nlist, 0);

    for (int it = 0; it < iters; ++it) {
        // assign
        std::fill(counts.begin(), counts.end(), 0);
        if (threads <= 1) {
            for (size_t i = 0; i < n; ++i) {
                float best = numeric_limits<float>::infinity();
                int bestj = 0;
                for (int j = 0; j < nlist; ++j) {
                    float dist = l2sq(X + i*d, res.centers.data() + j*d, d);
                    if (dist < best) { best = dist; bestj = j; }
                }
                assign[i] = bestj;
                counts[bestj]++;
            }
        } else {
            vector<vector<pair<size_t,int>>> local_assign(threads);
            vector<thread> ts;
            atomic<bool> ok(true);
            for (int t = 0; t < threads; ++t) {
                ts.emplace_back([&,t]() {
                    size_t chunk = (n + threads - 1) / threads;
                    size_t s = t * chunk, e = min(n, s + chunk);
                    vector<pair<size_t,int>> tmp;
                    tmp.reserve(e > s ? (e - s) : 0);
                    for (size_t i = s; i < e; ++i) {
                        float best = numeric_limits<float>::infinity();
                        int bestj = 0;
                        for (int j = 0; j < nlist; ++j) {
                            float dist = l2sq(X + i*d, res.centers.data() + j*d, d);
                            if (dist < best) { best = dist; bestj = j; }
                        }
                        tmp.emplace_back(i, bestj);
                    }
                    local_assign[t].swap(tmp);
                });
            }
            for (auto& th : ts) th.join();
            for (int t = 0; t < threads; ++t) {
                for (auto& p : local_assign[t]) {
                    assign[p.first] = p.second;
                    counts[p.second]++;
                }
            }
        }

        // recompute centers
        vector<float> newc(nlist * d, 0.f);
        if (threads <= 1) {
            for (size_t i = 0; i < n; ++i) {
                int cid = assign[i];
                const float* xi = X + i*d;
                float*       cj = newc.data() + cid*d;
                for (int z = 0; z < d; ++z) cj[z] += xi[z];
            }
        } else {
            int T = threads;
            vector<vector<float>> partial(T, vector<float>(nlist*d, 0.f));
            vector<thread> ts;
            for (int t = 0; t < T; ++t) {
                ts.emplace_back([&,t]() {
                    size_t chunk = (n + T - 1) / T;
                    size_t s = t * chunk, e = min(n, s + chunk);
                    float* acc = partial[t].data();
                    for (size_t i = s; i < e; ++i) {
                        int cid = assign[i];
                        const float* xi = X + i*d;
                        float* cj = acc + cid*d;
                        for (int z = 0; z < d; ++z) cj[z] += xi[z];
                    }
                });
            }
            for (auto& th : ts) th.join();
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < nlist*d; ++j) newc[j] += partial[t][j];
            }
        }
        for (int j = 0; j < nlist; ++j) {
            float* cj = newc.data() + j*d;
            size_t c = counts[j];
            if (c == 0) {
                // empty cluster: keep old center (simple strategy)
                continue;
            }
            for (int z = 0; z < d; ++z) cj[z] /= static_cast<float>(c);
        }
        // fill empty clusters by copying old centers; or jitter (omitted)
        for (int j = 0; j < nlist; ++j) {
            if (counts[j] == 0) {
                // keep previous
                memcpy(newc.data() + j*d, res.centers.data() + j*d, sizeof(float)*d);
            }
        }
        res.centers.swap(newc);
    }
    res.assign.swap(assign);
    return res;
}

struct IVF {
    int d = 0;
    int nlist = 0;
    vector<float> centers;                  // nlist * d
    bool from_device = false;
    string dev_path;                        // 盘符（仅用于盘搜索）
    uint64_t dev_offset = 0;                // 偏移量（仅用于盘搜索）
    const vector<float>* base = nullptr;    // 仅用于构建阶段或非设备搜索
    vector<vector<uint32_t>> lists;         // IVF list
};

static IVF build_ivf(const FVecs& base, const vector<float>& centers, int nlist, int threads=1) {
    IVF ivf;
    ivf.d = base.d;
    ivf.nlist = nlist;
    ivf.centers = centers;
    ivf.base = &base.data;
    ivf.lists.assign(nlist, {});
    size_t N = base.n;
    int d = base.d;
    const float* X = base.data.data();

    vector<int> assign(N);
    if (threads <= 1) {
        for (size_t i = 0; i < N; ++i) {
            float best = numeric_limits<float>::infinity();
            int bestj = 0;
            for (int j = 0; j < nlist; ++j) {
                float dist = l2sq(X + i*d, ivf.centers.data() + j*d, d);
                if (dist < best) { best = dist; bestj = j; }
            }
            assign[i] = bestj;
        }
    } else {
        int T = threads;
        vector<vector<pair<size_t,int>>> local(T);
        vector<thread> ts;
        for (int t = 0; t < T; ++t) {
            ts.emplace_back([&,t]() {
                size_t chunk = (N + T - 1) / T;
                size_t s = t * chunk, e = min(N, s + chunk);
                vector<pair<size_t,int>> tmp;
                tmp.reserve(e > s ? (e - s) : 0);
                for (size_t i = s; i < e; ++i) {
                    float best = numeric_limits<float>::infinity();
                    int bestj = 0;
                    for (int j = 0; j < nlist; ++j) {
                        float dist = l2sq(X + i*d, ivf.centers.data() + j*d, d);
                        if (dist < best) { best = dist; bestj = j; }
                    }
                    tmp.emplace_back(i, bestj);
                }
                local[t].swap(tmp);
            });
        }
        for (auto& th: ts) th.join();
        for (int t = 0; t < threads; ++t) {
            for (auto& p : local[t]) assign[p.first] = p.second;
        }
    }
    // pack lists
    for (size_t i = 0; i < N; ++i) ivf.lists[assign[i]].push_back((uint32_t)i);
    return ivf;
}

struct SearchResult {
    vector<int>   labels;   // nq * k
    vector<float> dists;    // nq * k (L2^2)
};

static SearchResult ivf_search(
    const IVF& ivf,
    const FVecs& queries,
    int nprobe,
    int k,
    int threads=1,
    size_t io_batch=1024)
{
    int d = ivf.d;
    const float* Q = queries.data.data();
    size_t nq = queries.n;
    const float* Xmem = (!ivf.from_device && ivf.base) ? ivf.base->data() : nullptr;
    int nlist = ivf.nlist;

    SearchResult R;
    R.labels.assign(nq * k, -1);
    R.dists.assign(nq * k, numeric_limits<float>::infinity());

    auto search_one = [&](size_t qi, std::ifstream* pfin, std::vector<float>* pbuf) {
        const float* q = Q + qi*d;

        // 1) pick nprobe nearest centers
        vector<pair<float,int>> cand;
        cand.reserve(nlist);
        for (int j = 0; j < nlist; ++j) {
            float dist = l2sq(q, ivf.centers.data() + j*d, d);
            cand.emplace_back(dist, j);
        }
        if (nprobe < nlist) {
            nth_element(cand.begin(), cand.begin() + nprobe, cand.end(),
                        [](auto& a, auto& b){ return a.first < b.first; });
            cand.resize(nprobe);
        } else {
            sort(cand.begin(), cand.end(), [](auto& a, auto& b){ return a.first < b.first; });
        }

        // 2) scan lists & keep top-k
        // use max-heap of size <= k
        using P = pair<float,int>;
        vector<P> heap; heap.reserve(k+1);

        if(!ivf.from_device) {
            if (!Xmem) { /* 不应发生 */ return; }
            for (auto& [_, cid] : cand) {
                for (uint32_t idx : ivf.lists[cid]) {
                    const float* x = Xmem + (size_t)idx * d;
                    float dist = l2sq(q, x, d);
                    if ((int)heap.size() < k) {
                        heap.emplace_back(dist, (int)idx);
                        push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; }); // max-heap by dist
                    } else if (dist < heap.front().first) {
                        pop_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                        heap.back() = {dist, (int)idx};
                        push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                    }
                }
            }
        }
        else {
            // 设备路径：从块设备按批读
            ifstream& fin = *pfin;
            vector<float>& buf = *pbuf;      // 大小至少 io_batch*d

            for (auto& [_, cid] : cand) {
                const auto& ids = ivf.lists[cid];
                size_t M = ids.size();
                for (size_t s = 0; s < M; s += io_batch) {
                    size_t e = min(M, s + io_batch);

                    // 把本批的 id 拷出并排序，合并连续段以减少 seek
                    vector<uint32_t> batch(ids.begin() + s, ids.begin() + e);
                    sort(batch.begin(), batch.end());

                    size_t run_start = 0;
                    while (run_start < batch.size()) {
                        size_t run_end = run_start + 1;
                        // 找 [run_start, run_end) 的连续段
                        while (run_end < batch.size() && batch[run_end] == batch[run_end-1] + 1) {
                            ++run_end;
                        }
                        // 这一段是连续的：一次顺序 read
                        size_t first_id = batch[run_start];
                        size_t count    = run_end - run_start;

                        if (buf.size() < count * (size_t)d) buf.resize(count * (size_t)d);
                        if (!read_vectors_from_device_batch(fin, ivf.dev_offset, first_id, count, d, buf.data())) {
                            // 读失败就降级为逐条读（健壮性），一般不会发生
                            for (size_t t = 0; t < count; ++t) {
                                if (!read_vectors_from_device_batch(fin, ivf.dev_offset, batch[run_start+t], 1, d, buf.data())) {
                                    continue;
                                }
                                float dist = l2sq(q, buf.data(), d);
                                int idx_id = (int)batch[run_start+t];
                                if ((int)heap.size() < k) {
                                    heap.emplace_back(dist, idx_id);
                                    push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                } else if (dist < heap.front().first) {
                                    pop_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                    heap.back() = {dist, idx_id};
                                    push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                }
                            }
                        } else {
                            // 用批量读出的 buf 逐条计算距离
                            for (size_t t = 0; t < count; ++t) {
                                const float* x = buf.data() + t * d;
                                float dist = l2sq(q, x, d);
                                int idx_id = (int)(first_id + t);
                                if ((int)heap.size() < k) {
                                    heap.emplace_back(dist, idx_id);
                                    push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                } else if (dist < heap.front().first) {
                                    pop_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                    heap.back() = {dist, idx_id};
                                    push_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
                                }
                            }
                        }
                        run_start = run_end;
                    } // while runs
                } // for batches
            } // for cand lists
        }
        // 3) write results sorted ascending
        sort_heap(heap.begin(), heap.end(), [](const P& a, const P& b){ return a.first < b.first; });
        size_t got = min((size_t)k, heap.size());
        for (size_t i = 0; i < got; ++i) {
            R.dists[qi*k + i]  = heap[i].first;
            R.labels[qi*k + i] = heap[i].second;
        }
    };

    if (threads <= 1) {
        // 单线程：打开一个设备流 + 缓冲
        ifstream fin;
        vector<float> buf; buf.reserve(io_batch * (size_t)d);
        if (ivf.from_device) {
            fin.open(ivf.dev_path, ios::binary);
            if (!fin) { cerr << "open device failed: " << ivf.dev_path << "\n"; return R; }
        }
        for (size_t i = 0; i < nq; ++i) search_one(i, ivf.from_device ? &fin : nullptr, &buf);
    } else {
        // 多线程：每个线程一个 ifstream + 缓冲，避免竞争
        vector<thread> ts;
        for (int t = 0; t < threads; ++t) {
            ts.emplace_back([&,t]() {
                ifstream fin_local;
                vector<float> buf; buf.reserve(io_batch * (size_t)d);
                if (ivf.from_device) {
                    fin_local.open(ivf.dev_path, ios::binary);
                    if (!fin_local) { cerr << "open device failed in thread: " << ivf.dev_path << "\n"; return; }
                }
                size_t chunk = (nq + (size_t)threads - 1) / (size_t)threads;
                size_t s = (size_t)t * chunk, e = min(nq, s + chunk);
                for (size_t i = s; i < e; ++i) search_one(i, ivf.from_device ? &fin_local : nullptr, &buf);
            });
        }
        for (auto& th : ts) th.join();
    }
    return R;
}

static double recall_at_k(const SearchResult& R, const IVecs& gt, int k) {
    if (gt.n != R.labels.size() / k) return -1.0;
    size_t nq = gt.n;
    int kgt = gt.k;
    const int* G = gt.data.data();
    size_t hit = 0;
    for (size_t i = 0; i < nq; ++i) {
        // collect set of top-k returned
        std::unordered_set<int> ret;
        ret.reserve(k*2);
        for (int j = 0; j < k; ++j) {
            int id = R.labels[i*k + j];
            if (id >= 0) ret.insert(id);
        }
        // check overlap with groundtruth's top-kgt (we measure against min(k,kgt))
        int kk = min(k, kgt);
        for (int j = 0; j < kk; ++j) {
            if (ret.count(G[i*kgt + j])) hit++;
        }
    }
    return (double)hit / (double)(nq * min(k, gt.k));
}

struct Args {
    string path_learn, path_base, path_query, path_gt;
    int nlist = 4096;
    int nprobe = 16;
    int k = 10;
    int iters = 20;
    int threads = 1;
    int sample_learn = 200000; // if no learn provided, sample base
    bool     search_from_dev = false; // 仅搜索时从设备读取
    string   base_dev;                // 设备路径(/dev/nvme0n1)
    uint64_t base_offset = 0;         // base 向量在设备上的起始字节偏移
    int      base_d_raw = 0;          // 维度（用于计算偏移）
    size_t   io_batch = 1024;         // 搜索阶段每次批量读取的向量数
    bool     stage_base_to_dev = false; // kmeans之前把base向量存盘
    bool     stage_centers_to_dev = false; // 把训练出的 centers 写到设备
    bool     load_centers_from_dev = false; // 直接从设备加载 centers，跳过kmeans
    uint64_t centers_offset = 0;           // centers 起始字节偏移
};

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto need = [&](const char* name){ if (i+1>=argc){ cerr<<"Missing value for "<<name<<"\n"; exit(1);} return string(argv[++i]); };
        if (s == "--learn") a.path_learn = need("--learn");
        else if (s == "--base") a.path_base = need("--base");
        else if (s == "--query") a.path_query = need("--query");
        else if (s == "--gt") a.path_gt = need("--gt");
        else if (s == "--nlist") a.nlist = stoi(need("--nlist"));
        else if (s == "--nprobe") a.nprobe = stoi(need("--nprobe"));
        else if (s == "--k") a.k = stoi(need("--k"));
        else if (s == "--iters") a.iters = stoi(need("--iters"));
        else if (s == "--threads") a.threads = stoi(need("--threads"));
        else if (s == "--sample_learn") a.sample_learn = stoi(need("--sample_learn"));
        else if (s == "--search_from_dev") a.search_from_dev = true;
        else if (s == "--base_dev") a.base_dev = need("--base_dev");
        else if (s == "--base_offset") a.base_offset = stoull(need("--base_offset"));
        else if (s == "--base_d") a.base_d_raw = stoi(need("--base_d"));
        else if (s == "--io_batch") a.io_batch = stoull(need("--io_batch"));
        else if (s == "--stage_base_to_dev") a.stage_base_to_dev = true;
        else if (s == "--stage_centers_to_dev") a.stage_centers_to_dev = true;
        else if (s == "--load_centers_from_dev") a.load_centers_from_dev = true;
        else if (s == "--centers_offset") a.centers_offset = stoull(need("--centers_offset"));
        
        else if (s == "--help" || s=="-h") {
            cout <<
            "Usage: ivf_sift --base sift_base.fvecs --query sift_query.fvecs [--learn sift_learn.fvecs]\n"
            "                [--gt sift_groundtruth.ivecs] [--nlist 4096] [--nprobe 16]\n"
            "                [--k 10] [--iters 20] [--threads 1] [--sample_learn 200000]\n"
            "                 --search_from_dev --base_dev /dev/nvme0n1 --base_offset 0\n"
            "                 --base_d 128 [--io_batch 1024] --stage_base_to_dev\n"
            "                 --stage_centers_to_dev --load_centers_from_dev [--centers_offset ]\n";
            exit(0);
        }
        else {
            cerr << "Unknown arg: " << s << "\n";
            return false;
        }
    }
    if (a.path_base.empty() || a.path_query.empty()) {
        cerr << "Required: --base and --query\n";
        return false;
    }
    return true;
}

static FVecs make_learn_from_base(const FVecs& base, int maxn, uint64_t seed=1234) {
    FVecs L; L.d = base.d;
    int d = base.d;
    size_t N = base.n;
    size_t m = min<size_t>(maxn, N);
    L.n = m;
    L.data.resize(m * d);
    // simple uniform sample without replacement (reservoir)
    std::mt19937_64 rng(seed);
    vector<size_t> idx(N);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    for (size_t i = 0; i < m; ++i) {
        memcpy(L.data.data() + i*d, base.data.data() + idx[i]*d, sizeof(float)*d);
    }
    return L;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    // 1) load base & query
    auto t0 = chrono::high_resolution_clock::now();
    FVecs base, query, learn;
    if (!read_fvecs(args.path_base, base)) {
        cerr << "Failed to read base: " << args.path_base << "\n"; return 1;
    }
    if (!read_fvecs(args.path_query, query)) {
        cerr << "Failed to read query: " << args.path_query << "\n"; return 1;
    }
    if (!args.path_learn.empty()) {
        if (!read_fvecs(args.path_learn, learn)) {
            cerr << "Failed to read learn: " << args.path_learn << "\n"; return 1;
        }
    } else {
        cerr << "[Info] --learn not provided. Sampling " << args.sample_learn << " from base for k-means.\n";
        learn = make_learn_from_base(base, args.sample_learn);
    }
    if (base.d != query.d || base.d != learn.d) {
        cerr << "Dim mismatch among datasets.\n"; return 1;
    }
    auto t1 = chrono::high_resolution_clock::now();
    double t_load = chrono::duration<double>(t1 - t0).count();

    cerr << "Loaded: base n=" << base.n << " d=" << base.d
         << ", learn n=" << learn.n
         << ", query n=" << query.n
         << " (time " << t_load << " s)\n";

    // 2) 把base向量写到SSD上
    // 若未显式给 --base_d，则用实际维度填充，便于后续一致性
    if (args.stage_base_to_dev) {
        if (args.base_dev.empty()) {
        cerr << "[ABORT] --stage_base_to_dev 需要提供 --base_dev（如 /dev/nvme0n1）。\n";
        return 2;
        }

        if (args.base_d_raw <= 0) args.base_d_raw = base.d;
        if (args.base_d_raw != base.d) {
            cerr << "[ABORT] base 维度不一致：--base_d=" << args.base_d_raw
                << " 但实际 base.d=" << base.d << "\n";
            return 2;
        }

        auto tw0 = chrono::high_resolution_clock::now();
        bool ok = write_raw_floats_to_block(
            args.base_dev,
            args.base_offset,              // 起始字节偏移
            base.data.data(),              // 连续的 float 数组
            base.n,                        // 向量条数
            base.d                         // 维度
        );
        auto tw1 = chrono::high_resolution_clock::now();
        if (!ok) {
            cerr << "[ABORT] 写入设备失败： " << args.base_dev << "\n";
            return 2;
        }
        double t_wr = chrono::duration<double>(tw1 - tw0).count();
        double gb   = (double)base.n * (double)base.d * sizeof(float) / 1e9;
        cerr << "[Stage-BASE] wrote n=" << base.n << " d=" << base.d
            << " to " << args.base_dev << " @offset=" << args.base_offset
            << "  bytes=" << (size_t)base.n * (size_t)base.d * sizeof(float)
            << "  time=" << fixed << setprecision(3) << t_wr << " s"
            << "  bw≈" << setprecision(2) << (gb / max(t_wr, 1e-9)) << " GB/s\n";
    }
    // 3.1) 从盘上读kmeans结果
    vector<float> centers_loaded;
    bool use_loaded_centers = false;
    if (args.load_centers_from_dev) {
        if (args.base_dev.empty()) {
            cerr << "[ABORT] --load_centers_from_dev 需要 --base_dev\n"; return 2;
        }
        int nlist_for_centers = args.nlist;
        int d_for_centers     = base.d;

        auto tld0 = chrono::high_resolution_clock::now();
        if (!read_centers_from_block(args.base_dev, args.centers_offset,
                                    nlist_for_centers, d_for_centers, centers_loaded)) {
            cerr << "[ABORT] 读取 centers 失败\n"; return 2;
        }
        auto tld1 = chrono::high_resolution_clock::now();
        double t_ld = chrono::duration<double>(tld1 - tld0).count();
        cerr << "[Centers-LOAD] nlist=" << nlist_for_centers << " d=" << d_for_centers
            << " from " << args.base_dev << " @offset=" << args.centers_offset
            << " time=" << fixed << setprecision(3) << t_ld << " s\n";

        if (nlist_for_centers != args.nlist || d_for_centers != base.d) {
            cerr << "[ABORT] centers 与当前参数不一致：nlist(" << nlist_for_centers
                << " vs " << args.nlist << "), d(" << d_for_centers << " vs " << base.d << ")\n";
            return 2;
        }
        use_loaded_centers = true;
    }

    // 3.2) train kmeans
    vector<float> centers_to_use;
    double t_km = 0.0;
    if (!use_loaded_centers) {
        auto t2 = chrono::high_resolution_clock::now();
        auto km = kmeans_train(learn, args.nlist, args.iters, args.threads);
        auto t3 = chrono::high_resolution_clock::now();
        double t_km = chrono::duration<double>(t3 - t2).count();
        cerr << "kmeans done. nlist=" << args.nlist
            << ", iters=" << args.iters
            << " (time " << t_km << " s)\n";
            centers_to_use.swap(km.centers);

        // [可选] 训练后立刻把 centers 落盘
        if (args.stage_centers_to_dev) {
            if (args.base_dev.empty()) {
                cerr << "[ABORT] --stage_centers_to_dev 需要 --base_dev\n"; return 2;
            }
            auto tw0 = chrono::high_resolution_clock::now();
            bool ok = write_centers_to_block(args.base_dev, args.centers_offset, centers_to_use);
            auto tw1 = chrono::high_resolution_clock::now();
            if (!ok) { cerr << "[ABORT] 写 centers 失败\n"; return 2; }
            double t_wr = chrono::duration<double>(tw1 - tw0).count();
            double gb   = centers_to_use.size() * sizeof(float) / 1e9;
            cerr << "[Centers-SAVE] wrote nlist*d=" << centers_to_use.size()
                << " floats to " << args.base_dev
                << " @offset=" << args.centers_offset
                << " time=" << fixed << setprecision(3) << t_wr << " s"
                << " bw≈" << setprecision(2) << (gb / max(t_wr, 1e-9)) << " GB/s\n";
        }
    }
    else {
        centers_to_use.swap(centers_loaded);
    }

    // 4) build IVF
    auto t4 = chrono::high_resolution_clock::now();
    auto ivf = build_ivf(base, centers_to_use, args.nlist, args.threads);
    if (args.search_from_dev) {
        ivf.from_device = true;
        ivf.dev_path    = args.base_dev;
        ivf.dev_offset  = args.base_offset;

        ivf.base = nullptr; 
        vector<float>().swap(base.data); // 释放 base 向量的大块内存
    }
    auto t5 = chrono::high_resolution_clock::now();
    double t_build = chrono::duration<double>(t5 - t4).count();
    size_t total_links = 0;
    size_t nonempty = 0;
    for (int i = 0; i < ivf.nlist; ++i) {
        total_links += ivf.lists[i].size();
        if (!ivf.lists[i].empty()) nonempty++;
    }
    cerr << "IVF built. lists=" << ivf.nlist
         << ", nonempty=" << nonempty
         << ", total_postings=" << total_links
         << " (time " << t_build << " s)\n";

    // 4) search
    auto t6 = chrono::high_resolution_clock::now();
    auto R = ivf_search(ivf, query, args.nprobe, args.k, args.threads, args.io_batch);
    auto t7 = chrono::high_resolution_clock::now();
    double t_search = chrono::duration<double>(t7 - t6).count();
    double qps = (query.n > 0) ? (query.n / t_search) : 0.0;

    cerr << "Search done. nprobe=" << args.nprobe
         << ", topk=" << args.k
         << " (time " << t_search << " s, QPS=" << qps << ")\n";

    // 5) optional: recall
    if (!args.path_gt.empty()) {
        IVecs gt;
        if (!read_ivecs(args.path_gt, gt)) {
            cerr << "Failed to read gt: " << args.path_gt << "\n";
        } else {
            int kk = min(args.k, gt.k);
            double r = recall_at_k(R, gt, kk);
            if (r >= 0.0) cerr << "Recall@" << kk << " = " << fixed << setprecision(4) << r << "\n";
        }
    }

    // 6) output some results
    int show = min<size_t>(5, query.n);
    for (int i = 0; i < show; ++i) {
        for (int j = 0; j < args.k; ++j) {
            if (j) cout << ",";
            cout << R.labels[i*args.k + j] << ":" << R.dists[i*args.k + j];
        }
        cout << "\n";
    }

    return 0;
}

