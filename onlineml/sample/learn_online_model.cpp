#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <onlineml/common/classifier.h>
#include <onlineml/learner/perceptron.h>
#include <onlineml/learner/averaged_perceptron.h>
#include <onlineml/util/string_proc.h>
#include "arg.h"

int main(int argc, char* argv[]) {
    ArgParser argparser;
    argparser.parse_args(argc, argv);

    int epoch = argparser.epoch;
    std::string modelfile = argparser.model_file;
    std::string alg = argparser.alg;

    printf("epoch:%d\n", epoch);
    printf("modelfile:%s\n", modelfile.c_str());
    printf("algoirthm:%s\n", alg.c_str());
    printf("trainfile:%s\n", argparser.train_file.c_str());
    if (argparser.test_file != "") {
        printf("testfile:%s\n", argparser.test_file.c_str());
    }

    Learner* p = argparser.learner;
    std::ifstream ifs(argparser.train_file.c_str());
    std::string line;

    std::vector<std::string> y;
    std::vector< std::vector< std::pair<std::string, float> > > x;

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

        y.push_back(label);
        x.push_back(fv);
    }

    for (int t=0; t<epoch; t++) {
        printf("epoch:%d/%d\n", t+1, epoch);
        p->fit(x, y);
    }

    p->save(modelfile.c_str());

    Classifier cls;
    cls.load(modelfile.c_str());

    if (argparser.test_file == "") {
        exit(1);
    }
    std::ifstream ifs2(argparser.test_file.c_str());

    int num_corr = 0;
    int num_total = 0;
    float accuracy = 0.;

    while(getline(ifs2, line)) {
        if (ifs2.fail()) {
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

//        int pred_ = p->predict(fv);
//        std::string pred = p->id2label(pred_);
        int pred_ = cls.predict(fv);
        std::string pred = cls.id2label(pred_);

//        int true_ = p->label2id(label);
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
