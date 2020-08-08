#ifndef INCLUDED_LEARNER
#define INCLUDED_LEARNER

#ifndef INCLUDED_DICT
#include <onlineml/common/dict.hpp>
#endif

class Learner {
    public:
        Dict features;
        Dict labels;
        std::vector< std::vector<float> > weight;

        Learner(){
            this->features = Dict();
            this->labels = Dict();
        };
        virtual ~Learner(){};
        virtual void expand_params(size_t i){};
        virtual void expand_params(size_t i, size_t j){};

        void fit(std::vector< std::vector< std::pair<std::string, float> > >& x,
                std::vector<std::string>& y);
        void fit(std::vector< std::vector< std::pair<size_t, float> > >& x,
                std::vector<size_t>& y);
        virtual void terminate_fitting() {};

        size_t predict(std::vector< std::pair<std::string, float> >& x);
        const char* id2label(size_t id);
        size_t label2id(std::string label);

        virtual void update_weight(size_t true_id, size_t argmax,
                std::vector< std::pair<size_t, float> >& fv){};
        virtual void save(const char*){};
};

const char* Learner::id2label(size_t id) {
    return this->labels.get_elem(id).c_str();
};

size_t Learner::label2id(std::string label) {
    return this->labels.get_id(label);
};

void Learner::fit(std::vector< std::vector< std::pair<size_t, float> > >& x,
        std::vector<size_t>& y) {
    size_t num_data = x.size();

    for (size_t i=0; i < num_data; ++i) {
        size_t yid = y[i];
        
        std::vector< std::pair<size_t, float> > fv;
        fv = x[i];

        size_t argmax = -1;
        float max = -1e50;

        for (size_t j=0; j < this->labels.size(); ++j) {
            this->expand_params(j);
            std::vector<float>* w_j = &this->weight[j];

            float dot = 0.;

            /* iterate over features */
            for (size_t _f=0; _f < fv.size(); ++_f) {
                size_t fid = fv[_f].first;
                float val = fv[_f].second;

                this->expand_params(j, fid);
                dot += val * (*w_j)[fid];
            }

//            if (dot >= max) {
            if (dot > max) {
                argmax = j;
                max = dot;
            }
        }

        if (argmax != yid) {
            this->update_weight(yid, argmax, fv);
        }
    }
}


void Learner::fit(std::vector< std::vector< std::pair<std::string, float> > >& x,
        std::vector<std::string>& y) {
    size_t num_data = x.size();

    for (size_t i=0; i < num_data; i++) {

        if (!this->labels.has_elem(y[i])) {
            this->labels.add_elem(y[i]);
        }
        size_t yid = this->labels.get_id(y[i]);

        this->expand_params(yid);
        
        std::vector< std::pair<size_t, float> > fv;
        fv = std::vector< std::pair<size_t, float> >(x[i].size());
        for (size_t _f = 0; _f < x[i].size(); ++_f) {
            std::string ft = x[i][_f].first;
            float val = x[i][_f].second;

            if (!this->features.has_elem(ft)) {
                this->features.add_elem(ft);
            }
            size_t fid = this->features.get_id(ft);
            std::pair<size_t, float> ftval = std::make_pair(fid, val);
            fv[_f] = ftval;
        }

        size_t argmax = -1;
        float max = -1e5;

        for (size_t j=0; j < this->labels.size(); j++) {
            std::vector<float>* w_j = &this->weight[j];
            if (w_j == NULL) {
                printf("NULL\n");
                printf("label %zu %s\n", j, this->labels.elems[j].c_str());
            }

            float dot = 0.;

            /* iterate over features */
            for (size_t _f=0; _f < fv.size(); _f++) {
//                size_t fid = fv[_f].first;
                size_t fid = fv[_f].first;
                float val = fv[_f].second;

                this->expand_params(j, fid);
                if ((*w_j).size() <= fid) {
                    printf("out of index\n");
                }
                dot += val * (*w_j)[fid];
            }

            if (dot >= max) {
                argmax = j;
                max = dot;
            }
        }

        if (argmax != yid) {
            this->update_weight(yid, argmax, fv);
        }
    }
}

size_t Learner::predict(std::vector< std::pair<std::string, float> >& x) {
    size_t argmax = -1;
    float max = -1e5;
    for (size_t j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
        for (size_t i=0; i < x.size(); ++i) {
            std::string ft = x[i].first;
            float val = x[i].second;

            if (!this->features.has_elem(ft)) {
                continue;
            }
            size_t fid = this->features.get_id(ft);
            dot += this->weight[j][fid] * val;
        }

        if (dot >= max) {
            max = dot;
            argmax = j;
        }

    }
    if (int(argmax) == -1) {
        printf("#label:%zu\n", this->labels.size());
    }
    return argmax;
}


#endif
