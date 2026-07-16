// extract_hits.cpp — one-pass extraction of the July-beam QA branches from an
// official n_TOF processed root file into flat .npy arrays ("hit cache").
//
// Motivation: the 6 read-pass python scripts each re-read the 10-18 GB root file
// with uproot and re-sort the hits, then run per-bunch pairing loops. Everything
// downstream only needs 9 small branches per hit tree, sorted by (BunchNumber,
// tof). This program reads each tree once (trees processed in parallel threads),
// sorts, and dumps per-branch .npy files that the python side memory-maps.
//
// Output (dir <out>/):
//   <TREE>_bunch.npy    int32     (sorted primary key)
//   <TREE>_tof.npy      float64   (sorted secondary key, within bunch)
//   <TREE>_tflash.npy   float64
//   <TREE>_amp.npy      float32
//   <TREE>_area.npy     float32
//   <TREE>_fwhm.npy     float32
//   <TREE>_detn.npy     uint8
//   <TREE>_satu.npy     uint8     (satuflag != 0)
//   <TREE>_pileup.npy   uint8     (pileup1  != 0)
//   index_bunch.npy     int32     \  per-bunch intensity, sorted by bunch
//   index_intensity.npy float64   /  (falls back to PKUP if index is empty)
//   meta.json           entry counts + provenance
//
// Usage: extract_hits <run.root> [out_dir]     (default out: <dir>/hitcache/<stem>)
// Build: make  (see Makefile; needs root-config in PATH)

#include <TFile.h>
#include <TTree.h>
#include <ROOT/TThreadExecutor.hxx>
#include <TROOT.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

static const std::vector<std::string> HIT_TREES = {
    "WALA", "WALB", "WALC", "WALD", "PSSA", "PSSB", "PSSC", "PSSD"};
// at most this many trees in flight: full-tree buffers are ~1.5-2.5 GB each
static const int MAX_WORKERS = 4;

// ---------------------------------------------------------------- npy writing
static void write_npy(const std::string& path, const void* data, size_t n,
                      const char* descr, size_t itemsize) {
    std::string header = std::string("{'descr': '") + descr +
                         "', 'fortran_order': False, 'shape': (" +
                         std::to_string(n) + ",), }";
    size_t unpadded = 10 + header.size() + 1;           // magic+ver+len + dict + \n
    size_t pad = (64 - unpadded % 64) % 64;
    header += std::string(pad, ' ');
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "FATAL: cannot open %s for writing\n", path.c_str()); exit(1); }
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hlen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data), n * itemsize);
    if (!f) { fprintf(stderr, "FATAL: short write on %s\n", path.c_str()); exit(1); }
}

template <typename T>
static std::vector<T> gather(const std::vector<T>& v, const std::vector<int64_t>& idx) {
    std::vector<T> out(v.size());
    for (size_t i = 0; i < idx.size(); ++i) out[i] = v[idx[i]];
    return out;
}

// ------------------------------------------------------------- per-tree work
static int64_t process_tree(const std::string& file, const std::string& tname,
                            const std::string& out_dir) {
    auto t0 = std::chrono::steady_clock::now();
    TFile f(file.c_str(), "READ");
    TTree* tree = f.Get<TTree>(tname.c_str());
    if (!tree) { fprintf(stderr, "FATAL: tree %s not found\n", tname.c_str()); exit(1); }
    const int64_t n = tree->GetEntries();

    tree->SetBranchStatus("*", 0);
    int32_t b_bunch, b_detn, b_satu, b_pileup;
    double b_tof, b_tflash;
    float b_amp, b_area, b_fwhm;
    struct { const char* name; void* addr; } wire[] = {
        {"BunchNumber", &b_bunch}, {"detn", &b_detn}, {"satuflag", &b_satu},
        {"pileup1", &b_pileup},    {"tof", &b_tof},   {"tflash", &b_tflash},
        {"amp", &b_amp},           {"area", &b_area}, {"fwhm", &b_fwhm}};
    for (auto& w : wire) {
        tree->SetBranchStatus(w.name, 1);
        if (tree->SetBranchAddress(w.name, w.addr) < 0) {
            fprintf(stderr, "FATAL: branch %s missing in %s\n", w.name, tname.c_str());
            exit(1);
        }
    }

    std::vector<int32_t> bunch(n);
    std::vector<double> tof(n), tflash(n);
    std::vector<float> amp(n), area(n), fwhm(n);
    std::vector<uint8_t> detn(n), satu(n), pileup(n);
    for (int64_t i = 0; i < n; ++i) {
        tree->GetEntry(i);
        bunch[i] = b_bunch;
        tof[i] = b_tof;
        tflash[i] = b_tflash;
        amp[i] = b_amp;
        area[i] = b_area;
        fwhm[i] = b_fwhm;
        detn[i] = static_cast<uint8_t>(b_detn);
        satu[i] = b_satu != 0;
        pileup[i] = b_pileup != 0;
    }
    f.Close();

    // stable sort by (bunch, tof) — matches np.lexsort((tof, bunch))
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        if (bunch[a] != bunch[b]) return bunch[a] < bunch[b];
        return tof[a] < tof[b];
    });

    const std::string p = out_dir + "/" + tname + "_";
    write_npy(p + "bunch.npy",  gather(bunch, idx).data(),  n, "<i4", 4);
    write_npy(p + "tof.npy",    gather(tof, idx).data(),    n, "<f8", 8);
    write_npy(p + "tflash.npy", gather(tflash, idx).data(), n, "<f8", 8);
    write_npy(p + "amp.npy",    gather(amp, idx).data(),    n, "<f4", 4);
    write_npy(p + "area.npy",   gather(area, idx).data(),   n, "<f4", 4);
    write_npy(p + "fwhm.npy",   gather(fwhm, idx).data(),   n, "<f4", 4);
    write_npy(p + "detn.npy",   gather(detn, idx).data(),   n, "|u1", 1);
    write_npy(p + "satu.npy",   gather(satu, idx).data(),   n, "|u1", 1);
    write_npy(p + "pileup.npy", gather(pileup, idx).data(), n, "|u1", 1);

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  %s: %lld hits in %.1f s\n", tname.c_str(), static_cast<long long>(n), dt);
    fflush(stdout);
    return n;
}

// per-bunch intensity from 'index', falling back to PKUP when index is empty
static int64_t write_index(const std::string& file, const std::string& out_dir) {
    TFile f(file.c_str(), "READ");
    const char* src = "index";
    TTree* tree = f.Get<TTree>(src);
    if (!tree || tree->GetEntries() == 0) {
        src = "PKUP";
        tree = f.Get<TTree>(src);
        if (!tree) { fprintf(stderr, "FATAL: neither index nor PKUP usable\n"); exit(1); }
    }
    const int64_t n = tree->GetEntries();
    int32_t b_bunch; float b_int;
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("BunchNumber", 1);
    tree->SetBranchStatus("PulseIntensity", 1);
    tree->SetBranchAddress("BunchNumber", &b_bunch);
    tree->SetBranchAddress("PulseIntensity", &b_int);
    std::vector<int32_t> bunch(n);
    std::vector<double> inten(n);
    for (int64_t i = 0; i < n; ++i) {
        tree->GetEntry(i);
        bunch[i] = b_bunch;
        inten[i] = b_int;
    }
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int64_t a, int64_t b) { return bunch[a] < bunch[b]; });
    write_npy(out_dir + "/index_bunch.npy", gather(bunch, idx).data(), n, "<i4", 4);
    write_npy(out_dir + "/index_intensity.npy", gather(inten, idx).data(), n, "<f8", 8);
    printf("  index (%s): %lld bunches\n", src, static_cast<long long>(n));
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <run.root> [out_dir]\n", argv[0]);
        return 1;
    }
    const std::string run_file = argv[1];
    std::string stem = run_file.substr(run_file.find_last_of('/') + 1);
    stem = stem.substr(0, stem.rfind(".root"));
    std::string dir = run_file.substr(0, run_file.find_last_of('/') + 1);
    const std::string out_dir =
        (argc > 2) ? argv[2] : dir + "hitcache/" + stem;
    mkdir((dir + "hitcache").c_str(), 0775);
    mkdir(out_dir.c_str(), 0775);

    ROOT::EnableThreadSafety();
    auto t0 = std::chrono::steady_clock::now();
    printf("extracting %s -> %s\n", run_file.c_str(), out_dir.c_str());

    std::vector<int64_t> counts(HIT_TREES.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int w = 0; w < MAX_WORKERS; ++w)
        pool.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < HIT_TREES.size();)
                counts[i] = process_tree(run_file, HIT_TREES[i], out_dir);
        });
    for (auto& t : pool) t.join();
    int64_t n_index = write_index(run_file, out_dir);

    std::string meta = "{\n  \"run_file\": \"" + run_file + "\",\n  \"n_bunches\": " +
                       std::to_string(n_index) + ",\n  \"trees\": {";
    for (size_t i = 0; i < HIT_TREES.size(); ++i)
        meta += (i ? ", " : "") + std::string("\"") + HIT_TREES[i] +
                "\": " + std::to_string(counts[i]);
    meta += "},\n  \"sorted_by\": \"(bunch, tof)\"\n}\n";
    std::ofstream(out_dir + "/meta.json") << meta;

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("done: %lld hits total in %.1f s\n",
           static_cast<long long>(std::accumulate(counts.begin(), counts.end(), int64_t{0})),
           dt);
    return 0;
}
