// Directed double-edge swaps: exact in/out degrees; existing self-loops fixed.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

std::vector<uint32_t> read(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() % 4 != 0) throw std::runtime_error("invalid input");
    std::vector<uint32_t> result(static_cast<size_t>(file.tellg()) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), result.size() * 4)) throw std::runtime_error("read failed");
    return result;
}
uint64_t key(uint32_t row, uint32_t col) { return (uint64_t(row) << 32) | col; }
int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("rows.bin cols.bin output.bin seed swaps_per_edge");
        auto rows = read(argv[1]), cols = read(argv[2]);
        if (rows.size() != cols.size() || rows.empty()) throw std::runtime_error("shape mismatch");
        std::unordered_set<uint64_t> edges;
        edges.max_load_factor(0.7f);
        edges.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            if (!edges.insert(key(rows[i], cols[i])).second) throw std::runtime_error("duplicate input edge");
        std::mt19937_64 rng(std::stoull(argv[4]));
        std::uniform_int_distribution<size_t> sample(0, rows.size() - 1);
        const uint64_t goal = rows.size() * std::stoull(argv[5]);
        uint64_t accepted = 0, attempted = 0;
        while (accepted < goal && attempted < goal * 100) {
            ++attempted;
            size_t i = sample(rng), j = sample(rng);
            auto a = rows[i], b = cols[i], c = rows[j], d = cols[j];
            if (a == c || b == d || a == b || c == d || a == d || c == b) continue;
            if (edges.count(key(a, d)) || edges.count(key(c, b))) continue;
            edges.erase(key(a, b)); edges.erase(key(c, d));
            edges.insert(key(a, d)); edges.insert(key(c, b));
            cols[i] = d; cols[j] = b;
            ++accepted;
        }
        if (accepted < goal) throw std::runtime_error("insufficient accepted swaps");
        std::ofstream out(argv[3], std::ios::binary);
        if (!out.write(reinterpret_cast<char*>(cols.data()), cols.size() * 4)) throw std::runtime_error("write failed");
        std::cout << "{\"acceptedSwaps\":" << accepted << ",\"attemptedSwaps\":" << attempted << "}" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
