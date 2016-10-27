#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#ifndef INCLUDED_PERCEPTRON
#define INCLUDED_PERCEPTRON

#ifndef INCLUDED_LEARNER
#include <onlineml/learner/learner.hpp>
#endif

#ifndef INCLUDED_DICT
#include <onlineml/common/dict.hpp>
#endif

class Perceptron: public Learner {
    public:
        Perceptron(): Learner(){};
        ~Perceptron(){};
        void expand_params(size_t yid);
        void expand_params(size_t yid, size_t fid);
        void update_weight(size_t true_id, size_t argmax, std::vector< std::pair<size_t, float> >& fv);
        void save(const char*);
};

void Perceptron::expand_params(size_t yid) {
    for (size_t k=this->weight.size(); k <= yid; ++k) {
        std::vector<float> w;
        this->weight.push_back(w);
    }
}

void Perceptron::expand_params(size_t yid, size_t fid) {
    for (size_t k=this->weight[yid].size(); k <= fid; ++k) {
        this->weight[yid].push_back(0.);
    }
}

void Perceptron::update_weight(size_t true_id, size_t argmax,
        std::vector< std::pair<size_t, float> >& fv) {

    std::vector<float>* w_t = &this->weight[true_id];
    std::vector<float>* w_f = &this->weight[argmax];
    for (size_t i=0; i < fv.size(); ++i) {
        size_t fid = fv[i].first;
        float val = fv[i].second;
        (*w_t)[fid] += val;
        (*w_f)[fid] -= val;
    }
};


void Perceptron::save(const char* filename) {
    FILE* fp = fopen(filename, "wb");

    std::string a = "perceptron";
    size_t a_size = a.size();
    fwrite(&a_size, sizeof(size_t), 1, fp);
    fwrite(a.data(), sizeof(char), a.size(), fp);

    size_t s = this->labels.size();
    fwrite(&s, sizeof(size_t), 1, fp);

    for (size_t i=0; i < this->labels.size(); i++) {
        std::string label = this->labels.elems[i];
        size_t s = label.size();
        fwrite(&s, sizeof(size_t), 1, fp);
        fwrite(label.data(), sizeof(char), label.size(), fp);
    }

    for (size_t i=0; i < this->labels.size(); i++) {
        size_t num_nonzero = 0;
        for (size_t j=0; j < this->features.size(); j++) {
            if (this->weight[i][j] != 0.) {
                num_nonzero += 1;
            }
        }
        fwrite(&num_nonzero, sizeof(size_t), 1, fp);

        for (size_t j=0; j < this->features.size(); j++) {
            if (this->weight[i][j] != 0.) {
                std::string ft = this->features.get_elem(j);
                size_t s = ft.size();
                fwrite(&s, sizeof(size_t), 1, fp);
                fwrite(ft.data(), sizeof(char), ft.size(), fp);
                fwrite(&this->weight[i][j], sizeof(float), 1, fp);
            }
        }
    }
    fclose(fp);
}

#endif
