#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <fstream>

#ifndef INCLUDED_PERCEPTRON
#define INCLUDED_PERCEPTRON
#include <onlineml/learner/learner.h>

#ifndef INCLUDED_DICT
#include <onlineml/common/dict.h>
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
        void fit2(std::vector< std::vector< std::pair<std::string, float> > > x,
                std::vector<std::string> y);
        int predict(std::map<std::string, float > x);
        int predict2(std::vector< std::pair<std::string, float > > x);
        const char* id2label(int id);
        int label2id(std::string);
        void save(const char*);
        void save2(const char*);
};

const char* Perceptron::id2label(int id) {
    return this->labels.get_elem(id).c_str();
};

int Perceptron::label2id(std::string label) {
    return this->labels.get_id(label);
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

void Perceptron::fit2(std::vector< std::vector< std::pair<std::string, float> > > x,
        std::vector<std::string> y) {
    int num_data = x.size();
    for (int i=0; i < num_data; i++) {

        if (!this->labels.has_elem(y[i])) {
            this->labels.add_elem(y[i]);
//            printf("add label\n");
        }
        int yid = this->labels.get_id(y[i]);

        for (int k=this->weight.size(); k <= yid; k++) {
            std::vector<float> w;
            this->weight.push_back(w);
        }
        
        std::vector< std::pair<int, float> > fv;
        for (size_t _f = 0; _f < x[i].size(); ++_f) {
            std::string ft = x[i][_f].first;
            float val = x[i][_f].second;

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

int Perceptron::predict2(std::vector< std::pair<std::string, float> > x) {
    int argmax = -1;
    float max = -1e5;
    for (int j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
        for (size_t i=0; i < x.size(); ++i) {
            std::string ft = x[i].first;
            float val = x[i].second;

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

void Perceptron::save2(const char* filename) {
    FILE* fp = fopen(filename, "wb");

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
