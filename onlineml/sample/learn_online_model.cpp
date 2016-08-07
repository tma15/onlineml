#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <onlineml/common/classifier.h>
#include <onlineml/learner/perceptron.h>
#include <onlineml/util/string_proc.h>

int main(int argc, char* argv[]) {
    /* parse command options */
    int result;
    int num_opt = 0;
    int epoch = 1;
    std::string modelfile;
    std::string alg;
    while((result=getopt(argc, argv, "e:m:"))!=-1){
        switch (result) {
            case 'm':
                modelfile = std::string(optarg);
                num_opt += 2;
                break;
            case 'e':
                epoch = atoi(optarg);
                num_opt += 2;
                break;
            case 'a':
                alg = std::string(optarg);
                num_opt += 2;
                break;
        }
    }
    printf("epoch:%d\n", epoch);
    printf("modelfile:%s\n", modelfile.c_str());
    printf("algoirthm:%s\n", alg.c_str());

    std::ifstream ifs(argv[num_opt+1]);
    num_opt += 1;
    std::string line;

//    Dict labels;
//    Dict features;
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

    Learner* p = new Perceptron();

    for (int t=0; t<epoch; t++) {
        printf("epoch:%d/%d\n", t+1, epoch);
        p->fit2(x, y);
    }

//    p->save("model");
    p->save2("model");

    Classifier cls;
    cls.load2("model");

    std::ifstream ifs2(argv[num_opt+1]);
    num_opt += 1;

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

//        int pred_ = p->predict2(fv);
//        std::string pred = p->id2label(pred_);
        int pred_ = cls.predict2(fv);
        std::string pred = cls.id2label(pred_);

//        int true_ = p->label2id(label);
        int true_ = cls.label2id(label);

        if (true_ == pred_) {
            num_corr += 1;
        }
        num_total += 1;
        accuracy = float(num_corr) / float(num_total);

        if (num_total % 1000==0) {
            printf("acc:%f (%d/%d) pred:%s(id:%d) true:%s(id:%d)\n",
                    accuracy, num_corr, num_total,
                    pred.c_str(), pred_, label.c_str(), true_);
        }
    }

    printf("acc:%f (%d/%d)\n", accuracy, num_corr, num_total);

    return 0;
}
