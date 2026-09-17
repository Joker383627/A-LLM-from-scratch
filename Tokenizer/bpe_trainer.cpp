#include <pybind11/pybind11.h>
#include<pybind11/stl.h>

#include<iostream>
#include<utility>
#include<vector>
#include<map>

namespace py = pybind11;

class BPETrainer{
    private:
    std::map<std::pair<int,int>, int> merge_table;
    std::map<int, std::vector<unsigned char>> vocab;

    public:
    BPETrainer(){
        for(int i = 0; i < 256; i++){
            vocab[i] = {static_cast<unsigned char>(i)};
        }
    };
    std::vector<int> train_bpe(
        const std::vector<int>& input_tokens,
        unsigned int vocab_size,
        unsigned int start_id
    );
    std::vector<int> merge_token(
        const std::vector<int>& tokens,
        std::pair<int,int> pair,
        unsigned int new_id
    );
    std::map<std::pair<int,int>,int> get_counts(
        const std::vector<int>& tokens
    );

    std::map<int, std::vector<unsigned char>> get_vocab();
    std::map<std::pair<int,int>, int> get_merge_table();
    void set_vocab(const std::map<int, std::vector<unsigned char>>& new_vocab);
    void set_merge_table(const std::map<std::pair<int,int>, int>& new_merge_table);
};

std::vector<int> BPETrainer::merge_token(
        const std::vector<int>& tokens,
        std::pair<int,int> pair,
        unsigned int new_id
    ){

        const int a = pair.first;
        const int b = pair.second;
        const int id = new_id;

        std::vector<int> new_tokens;
        new_tokens.reserve(tokens.size());

        std :: size_t i = 0;
        while(i < tokens.size()){
            if (i+1 < tokens.size() && tokens[i] == a && tokens[i+1] == b){
                new_tokens.push_back(id);
                i = i+2;
            }
            else{
                new_tokens.push_back(tokens[i]);
                i = i+1;
            }
        }
        return new_tokens;
    }

std::map<std::pair<int,int>,int> BPETrainer::get_counts(
        const std::vector<int>& tokens
    ){
        std::map<std::pair<int,int>,int> count_dict;

        std::size_t i = 0;
        while(i+1 < tokens.size()){
            std::pair<int,int> pair;
            pair.first = tokens[i];
            pair.second = tokens[i+1];

            count_dict[pair]++;
            i++;
        }
        return count_dict;
    }

std::vector<int> BPETrainer::train_bpe(
    const std::vector<int>& input_tokens,
    unsigned int vocab_size,
    unsigned int start_id
    ){
    
    if (vocab_size <= start_id){
        std::vector<int> r;
        r.push_back(-1);
        std::cerr<<"Too Small vocabulary size must be > 256"<<std::endl;
        return r;
    }

    std::cerr<<"C++ Trainer is Called\n";

    std::vector<int> tokens = input_tokens;
    
    int num_merges = vocab_size - start_id;
    
    for(int k = 0; k < num_merges ; k++){
        std::map<std::pair<int,int>,int> count_dict = get_counts(tokens);
        if (tokens.size() < 2 || count_dict.empty()) {
            break;
        }

        std::pair<int,int> max_pair;
        int max_count = 0;

        for(const auto& item : count_dict){
            if(item.second > max_count){
                max_count = item.second;
                max_pair = item.first;
            }
        }

        int new_id = start_id + k;

        merge_table[max_pair] = new_id;
        vocab[new_id] = vocab[max_pair.first];

        vocab[new_id].insert(
            vocab[new_id].end(),
            vocab[max_pair.second].begin(),
            vocab[max_pair.second].end()
        );

        tokens = merge_token(tokens,max_pair,new_id);
    };
    return tokens;
}

std::map<std::pair<int,int>,int> BPETrainer::get_merge_table(){
    return merge_table;
}
std::map<int,std::vector<unsigned char>> BPETrainer::get_vocab(){
    return vocab;
}
void BPETrainer::set_vocab(
    const std::map<int, std::vector<unsigned char>>& new_vocab
    ){
    vocab = new_vocab;
    }
void BPETrainer::set_merge_table(
    const std::map<std::pair<int,int>, int>& new_merge_table
    ){
    merge_table = new_merge_table;
    }

PYBIND11_MODULE(bpe_trainer, m) {
    m.doc() = "C++ BPE Trainer";

    py::class_<BPETrainer>(m, "BPETrainer")
        .def(py::init<>())
        .def("train_bpe", &BPETrainer::train_bpe)
        .def("merge_token", &BPETrainer::merge_token)
        .def("get_counts", &BPETrainer::get_counts)
        .def("get_vocab", &BPETrainer::get_vocab)
        .def("get_merge_table", &BPETrainer::get_merge_table)
        .def("set_merge_table",&BPETrainer::set_merge_table)
        .def("set_vocab",&BPETrainer::set_vocab);
}