#include <fstream>
#include <sstream>
#include <iostream>

#include <onlineml/common/dict.h>

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
        int predict(std::vector< std::pair<std::string, float> > x);
        const char* id2label(int id);
        int label2id(std::string label);
};

Classifier::Classifier() {
    this->features = Dict();
    this->labels = Dict();
}

void Classifier::load(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    int num_label;
    int num_feature;

    int a_size;
    fread(&a_size, sizeof(int), 1, fp);

    std::string algorithm;
    for (int k=0; k < a_size; ++k) {
        char ch;
        fread(&ch, sizeof(char), 1, fp);
        algorithm.push_back(ch);
    }
    printf("loaded algorithm:%s\n", algorithm.c_str());


    fread(&num_label, sizeof(int), 1, fp);
    fread(&num_feature, sizeof(int), 1, fp);

    for (int i=0; i < num_label; ++i) {
        int label_size;
        fread(&label_size, sizeof(int), 1, fp);
        std::string label;

        for (int k=0; k < label_size; ++k) {
            char ch;
            fread(&ch, sizeof(char), 1, fp);
            label.push_back(ch);
        }

        this->labels.add_elem(label);

        std::vector<float> w;
        this->weight.push_back(w);
    }

//    printf("#feature:%d\n", num_feature);
    for (int i=0; i < num_label; ++i) {
        for (int j=0; j < num_feature; ++j) {
            int ft_size = 0;
            fread(&ft_size, sizeof(int), 1, fp);

            std::string ft;
            for (int k=0; k < ft_size; ++k) {
                char ch;
                fread(&ch, sizeof(char), 1, fp);
                ft.push_back(ch);
            }

            float w;
            fread(&w, sizeof(float), 1, fp);

            if (!this->features.has_elem(ft)) {
                this->features.add_elem(ft);
            }
            int fid = this->features.get_id(ft);

            for (int k=this->weight[i].size(); k <= fid; k++) {
                this->weight[i].push_back(0.);
            }
            this->weight[i][fid] = w;
        }
    }
    fclose(fp);
}


const char* Classifier::id2label(int id) {
    return this->labels.get_elem(id).c_str();
};

int Classifier::label2id(std::string label) {
    return this->labels.get_id(label);
}

int Classifier::predict(std::vector< std::pair<std::string, float> > x) {
    int argmax = -1;
    float max = -1e5;
    for (int j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
        for (int i=0; i < x.size(); ++i) {
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

