
#ifndef INCLUDED_AVERAGED_PERCEPTRON
#define INCLUDED_AVERAGED_PERCEPTRON

#ifndef INCLUDED_LEARNER
#include <online/learner/learner.hpp>
#endif

#ifndef INCLUDED_DICT
#include <online/common/dict.hpp>
#endif 

class AveragedPerceptron: public Learner {
    private:
        size_t num_update;
        std::vector< std::vector<float> > weight_a;
    public:
        AveragedPerceptron();
        ~AveragedPerceptron(){};
        void expand_params(size_t yid);
        void expand_params(size_t yid, size_t fid);
        void terminate_fitting();
        void update_weight(size_t true_id, size_t argmax, std::vector< std::pair<size_t, float> >& fv);
        void save(const char*);
};

AveragedPerceptron::AveragedPerceptron(): Learner() {
    this->num_update = 1;
}

void AveragedPerceptron::expand_params(size_t yid) {
    for (size_t k=this->weight.size(); k <= yid; ++k) {
        std::vector<float> w;
        this->weight.push_back(w);
        this->weight_a.push_back(w);
    }
}

void AveragedPerceptron::expand_params(size_t yid, size_t fid) {
    for (size_t k=this->weight[yid].size(); k <= fid; ++k) {
        this->weight[yid].push_back(0.);
        this->weight_a[yid].push_back(0.);
    }
}


void AveragedPerceptron::update_weight(size_t true_id, size_t argmax, std::vector< std::pair<size_t, float> >& fv) {
    std::vector<float>* w_t = &this->weight[true_id];
    std::vector<float>* w_f = &this->weight[argmax];

    std::vector<float>* w_ta = &this->weight_a[true_id];
    std::vector<float>* w_fa = &this->weight_a[argmax];

    for (size_t i=0; i < fv.size(); ++i) {
        size_t fid = fv[i].first;
        float val = fv[i].second;
        (*w_t)[fid] += val;
        (*w_f)[fid] -= val;

        (*w_ta)[fid] += this->num_update * val;
        (*w_fa)[fid] -= this->num_update * val;
    }

    this->num_update += 1;

}

void AveragedPerceptron::terminate_fitting() {
    for (size_t i=0; i < this->weight.size(); ++i) {
        for (size_t j=0; j < this->weight[i].size(); ++j) {
            this->weight[i][j] -= this->weight_a[i][j] / this->num_update;
        }
    }
}

void AveragedPerceptron::save(const char* filename) {
    FILE* fp = fopen(filename, "wb");

    std::string a = "averaged_perceptron";
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
