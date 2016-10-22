#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <onlineml/common/classifier.hpp>
#include <onlineml/util/string_proc.hpp>

int main(int argc, char* argv[]) {
    std::string modelfile = std::string(argv[1]);
    std::string testfile = std::string(argv[2]);

    std::string line;

    Classifier cls;
    cls.load(modelfile.c_str());

    std::ifstream ifs(testfile.c_str());

    int num_corr = 0;
    int num_total = 0;
    float accuracy = 0.;

    while(getline(ifs, line)) {
        if (ifs.fail()) {
            std::cerr << "failed to open file:" << std::endl;
        }
        std::vector<std::string> elems;
        elems = split(line, ' ');

        std::vector< std::pair<std::string, float> > fv;
        std::string label = elems[0];
        for (int i=1; i < elems.size(); ++i) {

            std::vector<std::string> f_v = split(elems[i], ':');

            std::string f = f_v[0];
            float v = atof(f_v[1].c_str());

            fv.push_back(std::make_pair(f, v));
        }

        int pred_ = cls.predict(fv);
        std::string pred = cls.id2label(pred_);

        int true_ = cls.label2id(label);

        if (true_ == pred_) {
            num_corr += 1;
        }
        num_total += 1;
        accuracy = float(num_corr) / float(num_total);

        if (num_total % 1000==0) {
            printf("acc:%f (%d/%d) pred:%s (id:%d) true:%s (id:%d)\n",
                    accuracy, num_corr, num_total,
                    pred.c_str(), pred_, label.c_str(), true_);
        }
    }

    printf("acc:%f (%d/%d)\n", accuracy, num_corr, num_total);

    return 0;
}
