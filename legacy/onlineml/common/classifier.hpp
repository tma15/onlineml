#include <fstream>
#include <sstream>
#include <iostream>

#include <onlineml/common/dict.hpp>

template <typename List>
void split(const std::string& s, const std::string& delim, List& result) {
    result.clear();

    std::string::size_type pos = 0;

    while(pos != std::string::npos ) {
        std::string::size_type p = s.find(delim, pos);

        if(p == std::string::npos) {
            result.push_back(s.substr(pos));
            break;
        } else {
            result.push_back(s.substr(pos, p - pos));
        }
        pos = p + delim.size();
    }
}

class Classifier {
    
    public:
        Dict features;
        Dict labels;
        std::vector< std::vector<float> > weight;

        Classifier();
        void load(const char* filename);
        size_t predict(std::vector< std::pair<std::string, float> >& x);
        size_t predict(std::vector< std::pair<size_t, float> >& x);
        const char* id2label(size_t id);
        size_t label2id(std::string label);
};

Classifier::Classifier() {
    this->features = Dict();
    this->labels = Dict();
}

void Classifier::load(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    size_t num_label;
    size_t num_feature;

    size_t a_size;
    fread(&a_size, sizeof(size_t), 1, fp);

    std::string algorithm = "";
    for (size_t k=0; k < a_size; ++k) {
        char ch;
        fread(&ch, sizeof(char), 1, fp);
        algorithm.push_back(ch);
    }

    fread(&num_label, sizeof(size_t), 1, fp);

    for (size_t i=0; i < num_label; ++i) {
        size_t label_size;
        fread(&label_size, sizeof(size_t), 1, fp);
        std::string label;

        for (size_t k=0; k < label_size; ++k) {
            char ch;
            fread(&ch, sizeof(char), 1, fp);
            label.push_back(ch);
        }

        this->labels.add_elem(label);

        std::vector<float> w;
        this->weight.push_back(w);
    }

    for (size_t i=0; i < num_label; ++i) {
        fread(&num_feature, sizeof(size_t), 1, fp);
        for (size_t j=0; j < num_feature; ++j) {
            size_t ft_size = 0;
            fread(&ft_size, sizeof(size_t), 1, fp);

            std::string ft = "";
            for (size_t k=0; k < ft_size; ++k) {
                char ch;
                fread(&ch, sizeof(char), 1, fp);
                ft.push_back(ch);
            }

            float w = 0;
            fread(&w, sizeof(float), 1, fp);

            if (!this->features.has_elem(ft)) {
                this->features.add_elem(ft);
            }
            size_t fid = this->features.get_id(ft);

            for (size_t k=this->weight[i].size(); k <= fid; k++) {
                this->weight[i].push_back(0.);
            }
            this->weight[i][fid] = w;
        }
    }
    fclose(fp);
//    printf("\nloaded algorithm:%s\n", algorithm.c_str());
}


const char* Classifier::id2label(size_t id) {
    return this->labels.get_elem(id).c_str();
};

size_t Classifier::label2id(std::string label) {
    return this->labels.get_id(label);
}

size_t Classifier::predict(std::vector< std::pair<std::string, float> >& x) {
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
    if (argmax == -1) {
        printf("#label:%d\n", this->labels.size());
    }
//    printf("score:%f\n", max);
    return argmax;
}

size_t Classifier::predict(std::vector< std::pair<size_t, float> >& x) {
    size_t argmax = -1;
    float max = -1e5;
    for (size_t j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
        for (size_t i=0; i < x.size(); ++i) {
//            std::string ft = x[i].first;
            float val = x[i].second;

//            if (!this->features.has_elem(ft)) {
//                continue;
//            }
            size_t fid = x[i].first;
            if (fid >= this->weight[j].size()) {
                continue;
            }
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
//    printf("score:%f\n", max);
    return argmax;
}

