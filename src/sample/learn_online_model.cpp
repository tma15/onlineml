#include <stdio.h>

#include <vector>
#include <map>
#include <string>
#include <iostream>

#include <onlineml/learner/perceptron.h>

int main(int argc, char const* argv[]) {
    Learner* p = new Perceptron();

    std::vector< std::map<std::string, float> > x;
    std::vector< std::string > y;

    std::map<std::string, float> x1;
    x1.insert(std::map<std::string, float>::value_type("apple", 1.));
    x1.insert(std::map<std::string, float>::value_type("banana", 1.));
    x.push_back(x1);
    y.push_back("en");

    std::map<std::string, float> x2;
    x1.insert(std::map<std::string, float>::value_type("リンゴ", 1.));
    x1.insert(std::map<std::string, float>::value_type("バナナ", 1.));
    x.push_back(x2);
    y.push_back("ja");

    for (int t=0; t<3; t++) {
        p->fit(x, y);
    }

    int ret = p->predict(x2);
    std::cout << "label:" << ret << std::endl;
    const char* label = p->id2label(ret);
    printf("ret: %s (%d)\n", label, ret);
    return 0;
}
