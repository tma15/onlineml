#ifndef INCLUDED_LEARNER
#define INCLUDED_LEARNER

#ifndef INCLUDED_DICT
#include <onlineml/common/dict.h>
#endif

class Learner {
    private:
        virtual void expand_params(int i){};
        virtual void expand_params(int i, int j){};
    public:
        Dict features;
        Dict labels;
        std::vector< std::vector<float> > weight;

        Learner(){
            this->features = Dict();
            this->labels = Dict();
        };
        void fit(std::vector< std::vector< std::pair<std::string, float> > > x,
                std::vector<std::string> y);

        int predict(std::vector< std::pair<std::string, float> > x);
        const char* id2label(int id);
        int label2id(std::string label);

        virtual void update_weight(int true_id, int argmax,
                std::vector< std::pair<int, float> > fv){};
        virtual void save(const char*){};
};

const char* Learner::id2label(int id) {
    return this->labels.get_elem(id).c_str();
};

int Learner::label2id(std::string label) {
    return this->labels.get_id(label);
};

void Learner::fit(std::vector< std::vector< std::pair<std::string, float> > > x,
        std::vector<std::string> y) {
    int num_data = x.size();
    for (int i=0; i < num_data; i++) {

        if (!this->labels.has_elem(y[i])) {
            this->labels.add_elem(y[i]);
        }
        int yid = this->labels.get_id(y[i]);

        this->expand_params(yid);
//        for (int k=this->weight.size(); k <= yid; k++) {
//            std::vector<float> w;
//            this->weight.push_back(w);
//        }
        
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

                this->expand_params(j, fid);
//                for (int k=(*w_j).size(); k <= fid; k++) {
//                    (*w_j).push_back(0.);
//                }

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

int Learner::predict(std::vector< std::pair<std::string, float> > x) {
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


#endif
