#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include<iostream>

namespace py = pybind11;

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return (static_cast<std::size_t>(p.first) << 32)
             ^ static_cast<std::size_t>(p.second);
    }
};


class BPEEncoder {
private:

    // Ordered list of merges.
    // This is important because your Python implementation
    // applies merges in merge_table insertion order.
    std::vector<std::pair<int, int>> merge_pairs;

    std::vector<int> merge_ids;


public:

    BPEEncoder(
        const std::vector<std::pair<int, int>>& pairs,
        const std::vector<int>& ids
    )
        : merge_pairs(pairs), merge_ids(ids)
    {}


    std::vector<int> encode(const std::string& text) const {

        std::cerr << "C++ encoder called\n";

        // Python:
        //
        // raw_bytes = text.encode("utf-8")
        // tokens = list(raw_bytes)
        //
        std::vector<int> tokens;
        tokens.reserve(text.size());

        for (unsigned char c : text) {
            tokens.push_back(static_cast<int>(c));
        }


        // Exactly reproduce:
        //
        // for pair, new_id in self.merge_table.items():
        //     tokens = self.merge_tokens(tokens, pair, new_id)
        //
        for (std::size_t m = 0; m < merge_pairs.size(); ++m) {

            const int a = merge_pairs[m].first;
            const int b = merge_pairs[m].second;
            const int new_id = merge_ids[m];

            std::vector<int> new_tokens;
            new_tokens.reserve(tokens.size());

            std::size_t i = 0;

            while (i < tokens.size()) {

                if (
                    i + 1 < tokens.size() &&
                    tokens[i] == a &&
                    tokens[i + 1] == b
                ) {

                    new_tokens.push_back(new_id);
                    i += 2;

                } else {

                    new_tokens.push_back(tokens[i]);
                    i += 1;
                }
            }

            tokens.swap(new_tokens);
        }

        return tokens;
    }
};


PYBIND11_MODULE(bpe_encoder, m) {

    m.doc() = "C++ BPE encoder";

    py::class_<BPEEncoder>(m, "BPEEncoder")

        .def(
            py::init<
                const std::vector<std::pair<int, int>>&,
                const std::vector<int>&
            >()
        )

        .def(
            "encode",
            &BPEEncoder::encode
        );
}