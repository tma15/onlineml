#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <fstream>

#ifndef INCLUDED_PERCEPTRON
#define INCLUDED_PERCEPTRON
//#include "learner.h"
#include <onlineml/learner/learner.h>

#ifndef INCLUDED_DICT
//#include "include/dict.h"
#include <onlineml/dict.h>
#endif

class Perceptron: public Learner {
    public:
        Dict features;
        Dict labels;
        std::vector< std::vector<float> > weight;

        Perceptron(){
            this->features = Dict();
            this->labels = Dict();
        };
        virtual ~Perceptron() {
        };
        void fit(std::vector< std::map<std::string, float> > x, std::vector<std::string> y);
        int predict(std::map<std::string, float > x);
        const char* id2label(int id);
        void save(const char*);
};

const char* Perceptron::id2label(int id) {
    return this->labels.get_elem(id).c_str();
};

void Perceptron::fit(std::vector< std::map<std::string, float> > x,
        std::vector<std::string> y) {
    int num_data = x.size();
    for (int i=0; i < num_data; i++) {

        if (!this->labels.has_elem(y[i])) {
            this->labels.add_elem(y[i]);
        }
        int yid = this->labels.get_id(y[i]);

        for (int k=this->weight.size(); k <= yid; k++) {
            std::vector<float> w;
            this->weight.push_back(w);
        }
        
        std::vector< std::pair<int, float> > fv;
        for (std::map<std::string, float>::iterator it=x[i].begin();
                it!=x[i].end(); it++) {
            std::string ft = it->first;
            float val = it->second;

            if (!this->features.has_elem(ft)) {
                this->features.add_elem(ft);
            }
            int fid = this->features.get_id(ft);
            std::pair<int, float> ftval = std::make_pair(fid, val);
            fv.push_back(ftval);
        }

        int argmax = -1;
        float max = -1e5;

        for (int j=0; j < this->labels.size(); j++) {
            std::vector<float>* w_j = &this->weight[j];

            float dot = 0.;

            /* iterate over features */
            for (size_t _f=0; _f < fv.size(); _f++) {
                int fid = fv[_f].first;
                float val = fv[_f].second;

                for (int k=(*w_j).size(); k <= fid; k++) {
                    (*w_j).push_back(0.);
                }

                dot += val * (*w_j)[fid];
            }

            if (dot >= max) {
                argmax = j;
                max = dot;
            }
        }

        if (argmax != yid) {
            std::vector<float>* w_t = &this->weight[yid];
            std::vector<float>* w_f = &this->weight[argmax];
            for (size_t _f=0; _f < fv.size(); _f++) {
                int fid = fv[_f].first;
                float val = fv[_f].second;

                (*w_t)[fid] += 1. * val;
                (*w_f)[fid] -= 1. * val;
            }
        }
    }
}

int Perceptron::predict(std::map<std::string, float> x) {
    int argmax = -1;
    float max = -1e5;
    for (int j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
        for (std::map<std::string, float>::iterator it=x.begin();
                it!=x.end(); it++) {
            std::string ft = it->first;
            float val = it->second;

            if (!this->features.has_elem(ft)) {
                continue;
            }
            int fid = this->features.get_id(ft);
            dot += this->weight[j][fid] * val;
        }

        if (dot >= max) {
            max = dot;
            argmax = j;
        }

    }
    if (argmax == -1) {
        printf("#label:%d\n", this->labels.size());
    }
    return argmax;
}

void Perceptron::save(const char* filename) {
    std::ofstream ofs(filename);

    ofs << this->labels.size() << std::endl;
    for (int i=0; i < this->labels.size(); i++) {
        ofs << this->labels.elems[i] << std::endl;
    }

    for (int i=0; i < this->labels.size(); i++) {
        int num_nonzero = 0;
        for (int j=0; j < this->features.size(); j++) {
            if (this->weight[i][j] != 0.) {
                num_nonzero++;
            }
        }

        ofs << num_nonzero << std::endl;
        for (int j=0; j < this->features.size(); j++) {
            std::string ft = this->features.get_elem(j);
            if (this->weight[i][j]!=0.) {
                ofs << ft << "\t" << this->weight[i][j] << std::endl;
            }
        }
    }
}

#endif
