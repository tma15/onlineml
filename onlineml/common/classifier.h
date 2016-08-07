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
        void load2(const char* filename);
        int predict(std::map<std::string, float > x);
        int predict2(std::vector< std::pair<std::string, float> > x);
        const char* id2label(int id);
        int label2id(std::string label);
};

Classifier::Classifier() {
    this->features = Dict();
    this->labels = Dict();
}

void Classifier::load(const char* filename) {
    std::ifstream ifs(filename);

    std::string labelsize_str;
    getline(ifs, labelsize_str);

    int labelsize = atoi(labelsize_str.c_str());

    for (int i=0; i < labelsize; i++) {
        std::string label;
        getline(ifs, label);

        this->labels.add_elem(label);

        std::vector<float> w;
        this->weight.push_back(w);
    }


    for (int i=0; i < labelsize; i++) {
        std::string num_nonzero_str;
        getline(ifs, num_nonzero_str);
        int num_nonzero = atoi(num_nonzero_str.c_str());

//        std::cout << "NONZ:" << num_nonzero << std::endl;
        for (int j=0; j < num_nonzero; j++) {
            std::string ftval_str;
            getline(ifs, ftval_str);
            std::vector<std::string> sp;
            split(ftval_str, "\t", sp);
            std::string ft = sp[0];
            std::string valstr = sp[1];

            std::stringstream ss(valstr);
            float val;
            ss >> val;

            if (!this->features.has_elem(ft)) {
                this->features.add_elem(ft);
            }
            int fid = this->features.get_id(ft);

            for (int k=this->weight[i].size(); k <= fid; k++) {
                this->weight[i].push_back(0.);
            }
            this->weight[i][fid] = val;
        }
    }
}

void Classifier::load2(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    int num_label;
    int num_feature;

    fread(&num_label, sizeof(int), 1, fp);
    fread(&num_feature, sizeof(int), 1, fp);

    for (int i=0; i < num_label; ++i) {
        int label_size;
        int k = fread(&label_size, sizeof(int), 1, fp);
//        printf("label_size:%d k=%d\n", label_size, k);
        std::string label;

        for (int k=0; k < label_size; ++k) {
            char ch;
            fread(&ch, sizeof(char), 1, fp);
            label.push_back(ch);
        }
//        printf("label:%s\n", label.c_str());

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
                size_t sz = fread(&ch, sizeof(char), 1, fp);
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

//            printf("s:%d %d label:\"%s\"\n", ft_size, ft.size(), ft.data());
        }
    }
    fclose(fp);

//    std::ifstream ifs(filename);

//    std::string labelsize_str;
//    getline(ifs, labelsize_str);

//    int labelsize = atoi(labelsize_str.c_str());

//    for (int i=0; i < labelsize; i++) {
//        std::string label;
//        getline(ifs, label);

//        this->labels.add_elem(label);

//        std::vector<float> w;
//        this->weight.push_back(w);
//    }


//    for (int i=0; i < labelsize; i++) {
//        std::string num_nonzero_str;
//        getline(ifs, num_nonzero_str);
//        int num_nonzero = atoi(num_nonzero_str.c_str());

//        for (int j=0; j < num_nonzero; j++) {
//            std::string ftval_str;
//            getline(ifs, ftval_str);
//            std::vector<std::string> sp;
//            split(ftval_str, "\t", sp);
//            std::string ft = sp[0];
//            std::string valstr = sp[1];

//            std::stringstream ss(valstr);
//            float val;
//            ss >> val;

//            if (!this->features.has_elem(ft)) {
//                this->features.add_elem(ft);
//            }
//            int fid = this->features.get_id(ft);

//            for (int k=this->weight[i].size(); k <= fid; k++) {
//                this->weight[i].push_back(0.);
//            }
//            this->weight[i][fid] = val;
//        }
//    }
}


const char* Classifier::id2label(int id) {
    return this->labels.get_elem(id).c_str();
};

int Classifier::label2id(std::string label) {
    return this->labels.get_id(label);
}

int Classifier::predict(std::map<std::string, float> x) {
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


int Classifier::predict2(std::vector< std::pair<std::string, float> > x) {
    int argmax = -1;
    float max = -1e5;
    for (int j=0; j < this->labels.size(); j++) {
        /* iterate over features */
        float dot = 0;
//        for (std::map<std::string, float>::iterator it=x.begin();
//                it!=x.end(); it++) {
        for (int i=0; i < x.size(); ++i) {
//            std::string ft = it->first;
            std::string ft = x[i].first;
//            float val = it->second;
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

