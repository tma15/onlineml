#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#ifndef INCLUDED_PERCEPTRON
#define INCLUDED_PERCEPTRON

#ifndef INCLUDED_LEARNER
#include <onlineml/learner/learner.h>
#endif

#ifndef INCLUDED_DICT
#include <onlineml/common/dict.h>
#endif

class Perceptron: public Learner {
    private:
        void expand_params(int yid);
        void expand_params(int yid, int fid);
    public:
        Perceptron(): Learner(){};
        void update_weight(int true_id, int argmax, std::vector< std::pair<int, float> > fv);
        void save(const char*);
};

void Perceptron::expand_params(int yid) {
    for (int k=this->weight.size(); k <= yid; ++k) {
        std::vector<float> w;
        this->weight.push_back(w);
    }
}

void Perceptron::expand_params(int yid, int fid) {
    for (int k=this->weight[yid].size(); k <= fid; ++k) {
        this->weight[yid].push_back(0.);
    }
}

void Perceptron::update_weight(int true_id, int argmax,
        std::vector< std::pair<int, float> > fv) {

    std::vector<float>* w_t = &this->weight[true_id];
    std::vector<float>* w_f = &this->weight[argmax];
    for (int i=0; i < fv.size(); ++i) {
        int fid = fv[i].first;
        float val = fv[i].second;
        (*w_t)[fid] += 1. * val;
        (*w_f)[fid] -= 1. * val;
    }
};


void Perceptron::save(const char* filename) {
    FILE* fp = fopen(filename, "wb");

    std::string a = "perceptron";
    int a_size = a.size();
    fwrite(&a_size, sizeof(int), 1, fp);
    fwrite(a.data(), sizeof(char), a.size(), fp);

    int s = this->labels.size();
    fwrite(&s, sizeof(int), 1, fp);

    s = this->features.size();
    fwrite(&s, sizeof(int), 1, fp);

    for (int i=0; i < this->labels.size(); i++) {
        std::string label = this->labels.elems[i];
        int s = label.size();
        fwrite(&s, sizeof(int), 1, fp);
        fwrite(label.data(), sizeof(char), label.size(), fp);
    }

    for (int i=0; i < this->labels.size(); i++) {
        for (int j=0; j < this->features.size(); j++) {
            std::string ft = this->features.get_elem(j);
            int s = ft.size();
            fwrite(&s, sizeof(int), 1, fp);
            fwrite(ft.data(), sizeof(char), ft.size(), fp);
            fwrite(&this->weight[i][j], sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

#endif
