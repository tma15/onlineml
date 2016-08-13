#include <string>
#include <stdlib.h>

#include <onlineml/learner/learner.h>
#include <onlineml/learner/perceptron.h>
#include <onlineml/learner/averaged_perceptron.h>

class ArgParser {
    public:
        std::string train_file;
        std::string test_file;
        std::string model_file;
        std::string alg;
        int epoch;

        Learner* learner;

        ArgParser();
        void parse_args(int argc, char* argv[]);
        void print_help();
};

ArgParser::ArgParser() {
    this->epoch = 1;
    this->alg = "ap";
    this->train_file = "";
    this->test_file = "";
    this->model_file = "model";
};

void ArgParser::parse_args(int argc, char* argv[]) {
    int curr_pos = 1;

    for (int i=0; i < argc; ++i) {
        std::string optarg = std::string(argv[curr_pos]);
        if (optarg == "-h" || optarg == "--help") {
            this->print_help();
            exit(1);
        }
    }

    while (curr_pos < argc-2) {
        std::string optarg = std::string(argv[curr_pos]);
        if (optarg == "-e" || optarg == "--epoch") {
            curr_pos += 1;
            this->epoch = atoi(argv[curr_pos]);
        } else if (optarg == "-a" || optarg == "--alg") {
            curr_pos += 1;
            this->alg = std::string(argv[curr_pos]);
        } else if (optarg == "-m" || optarg == "--model") {
            curr_pos += 1;
            this->model_file = std::string(argv[curr_pos]);
        } else {
            printf("unknown option:%s\n", optarg.c_str());
            exit(1);
        }
        curr_pos += 1;
    }

    this->train_file = std::string(argv[curr_pos]);
    curr_pos += 1;

    if (curr_pos < argc) {
        this->test_file = std::string(argv[curr_pos]);
    }

    if (this->alg == "p") {
        this->learner = new Perceptron();
    } else if (this->alg == "ap") {
        this->learner = new AveragedPerceptron();
    } else {
        std::cerr << "invalid algorithm: " << this->alg << std::endl;
        exit(1);
    }
};

void ArgParser::print_help() {
    printf("train_online_model\n");
    printf("\n");
    printf("  -a, --alg     (p|ap)\n");
    printf("  -m, --model   model file\n");
    printf("  -e, --epoch   the number of epoch\n");
    printf("  -h, --help    print this help\n");
};
