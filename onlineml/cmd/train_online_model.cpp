#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef DEBUG
#include <time.h>
#endif

#include <pthread.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <onlineml/common/classifier.hpp>
#include <onlineml/common/dict.hpp>
#include <onlineml/learner/perceptron.hpp>
#include <onlineml/learner/averaged_perceptron.hpp>
#include <onlineml/util/string_proc.hpp>
#include "arg.hpp"

struct Trainer {
    pthread_t thread;
    size_t id;
    size_t epoch;
    Learner* learner;
    std::vector< std::vector< std::pair<size_t, float> > > x;
    std::vector<size_t> y;
};


void* run_trainer(void* arg) {
    struct Trainer* trainer = (struct Trainer*) arg;

#ifdef DEBUG
    clock_t start = clock();
#endif
    for (size_t i=0; i < trainer->epoch; ++i) {

#ifdef DEBUG
        clock_t start_i = clock();
#endif
        
        trainer->learner->fit(trainer->x, trainer->y);

#ifdef DEBUG
        clock_t end_i = clock();
        float time_i = (end_i - start_i) / CLOCKS_PER_SEC;
        printf("id:%d epoch %d/%d: fit thread id:%d %f sec.\n", trainer->id, i+1,
                trainer->epoch, trainer->id, time_i);
#else
        printf("id:%d epoch %d/%d: fit thread id:%d\n", trainer->id, i+1,
                trainer->epoch, trainer->id);
#endif

    }
    trainer->learner->terminate_fitting();
#ifdef DEBUG
    clock_t end = clock();
    float time = (end - start) / CLOCKS_PER_SEC;
    printf("thread %d: time %f sec. #data:%d\n", trainer->id, time, trainer->x.size());
#endif
    return NULL;
}

Learner* generate_learner(std::string alg) {
    Learner* learner;
    if (alg == "p") {
        learner = new Perceptron();
    } else if (alg == "ap") {
        learner = new AveragedPerceptron();
    } else {
        printf("alg:%s\n", alg.c_str());
        exit(1);
    }
    return learner;
}

Learner* ipm(std::vector< std::vector< std::pair<size_t, float> > >& x, std::vector<size_t>& y,
//        size_t epoch, ArgParser& argparser,
        size_t epoch,
        std::string alg,
        size_t num_parallel,
        Dict& labels, Dict& features) {
    printf("training model...\n");
    size_t num_train = x.size();
//    size_t num_parallel = argparser.num_parallel;
    size_t avg = num_train / num_parallel;
    std::vector<size_t> start_idx = std::vector<size_t>(num_parallel, 0);
    std::vector<size_t> end_idx = std::vector<size_t>(num_parallel, 0);
    for (size_t i=0; i < num_parallel-1; ++i) {
        size_t s = i * avg;
        size_t e = s + avg;
        start_idx[i] = s;
        end_idx[i] = e;
    }
    start_idx[num_parallel-1] = (num_parallel-1) * avg;
    end_idx[num_parallel-1] = num_train;

    std::vector<Trainer> threads;
    threads = std::vector<Trainer>(num_parallel);

    for (size_t i=0; i < num_parallel; ++i) {
#ifdef DEBUG
        printf("thread id:%d training idx: %d => %d\n", i, start_idx[i], end_idx[i]);
#endif

        Learner* learner_i = generate_learner(alg);
        learner_i->labels = labels;
        learner_i->features = features;

        std::vector< std::vector< std::pair<size_t, float> > > x_i;
        x_i = std::vector< std::vector< std::pair<size_t, float > > >(end_idx[i]-start_idx[i]);

        std::vector<size_t> y_i;
        y_i = std::vector<size_t>(end_idx[i]-start_idx[i]);

        size_t k = 0;
        for (size_t j=start_idx[i]; j < end_idx[i]; ++j) {
            x_i[k] = x[j];
            y_i[k] = y[j];
            k += 1;
        }

        threads[i].learner = learner_i;
        threads[i].x = x_i;
        threads[i].y = y_i;
        threads[i].id = i;
        threads[i].epoch = epoch;

        pthread_create(&threads[i].thread, NULL, run_trainer, &threads[i]);
    }

    Learner* avg_learner = generate_learner(alg);
    avg_learner->labels = labels;
    avg_learner->features = features;

    for (size_t i=0; i < num_parallel; ++i) {
        pthread_join(threads[i].thread, NULL);
    }

    /* merge learners */
    clock_t start = clock();
    for (size_t i=0; i < num_parallel; ++i) {
        Learner* learner_i = threads[i].learner;
        std::vector< std::vector<float> > w = learner_i->weight;

        size_t wsize = w.size();
        for (size_t yid=0; yid < wsize; ++yid) {

            size_t wysize = w[yid].size();
            for (size_t fid=0; fid < wysize; ++fid) {
                float w_ = w[yid][fid];
                if (w_ != 0.) {
                    avg_learner->weight[yid][fid] += (w_ / float(num_parallel));
                }
            }
        }
    }
    clock_t end = clock();
    std::cout << "time for merging: " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    return avg_learner;
};

int main(int argc, char* argv[]) {
    ArgParser argparser;
    argparser.parse_args(argc, argv);

    size_t epoch = argparser.epoch;
    std::string modelfile = argparser.model_file;
    std::string alg = argparser.alg;

    printf("epoch:%d\n", epoch);
    printf("modelfile:%s\n", modelfile.c_str());
    printf("algoirthm:%s\n", alg.c_str());
    printf("trainfile:%s\n", argparser.train_file.c_str());
    if (argparser.test_file != "") {
        printf("testfile:%s\n", argparser.test_file.c_str());
    }

    std::string line;

    std::vector<size_t> y;
    std::vector< std::vector< std::pair<size_t, float> > > x;

    Dict features;
    Dict labels;
    clock_t start_data = clock();

    read_data(argparser.train_file.c_str(),
            labels, features, y, x);

    clock_t end_data = clock();
    float time_data = (float)(end_data - start_data) / CLOCKS_PER_SEC;
    printf("reading finished! %f sec.\n", time_data);

    Learner* learner;
    if (argparser.num_parallel > 1) {
        learner = ipm(x, y, epoch, argparser.alg, argparser.num_parallel, labels, features);
    } else {
        learner = argparser.learner;
        learner->labels = labels;
        learner->features = features;
        for (size_t t=0; t<epoch; t++) {
            printf("epoch:%d/%d\n", t+1, epoch);
            learner->fit(x, y);
        }
    }

    learner->save(modelfile.c_str());

    if (argparser.test_file == "") {
        exit(0);
    }

    Classifier cls;
    cls.load(modelfile.c_str());
    std::ifstream ifs2(argparser.test_file.c_str());

    size_t num_corr = 0;
    size_t num_total = 0;
    float accuracy = 0.;

    while(getline(ifs2, line)) {
        if (ifs2.fail()) {
            std::cerr << "failed to open file:" << std::endl;
        }
        std::vector<std::string> elems;
        elems = split(line, ' ');

//        std::cout << line << std::endl;

        std::vector< std::pair<std::string, float> > fv;
        std::string label = elems[0];
        for (size_t i=1; i < elems.size(); ++i) {

            std::vector<std::string> f_v = split(elems[i], ':');

            std::string f = f_v[0];
            float v = atof(f_v[1].c_str());

            fv.push_back(std::make_pair(f, v));
        }

        size_t pred_ = cls.predict(fv);
        std::string pred = cls.id2label(pred_);

        size_t true_ = cls.label2id(label);

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
