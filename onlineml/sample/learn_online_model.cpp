#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <onlineml/util/string_proc.h>
#include <onlineml/learner/perceptron.h>

int main(int argc, char const* argv[]) {

    std::ifstream ifs("data");
    std::string line;

    std::vector<std::string> y;
    std::vector< std::vector< std::pair<std::string, float> > > x;

    while(getline(ifs, line)) {
        if (ifs.fail()) {
            std::cerr << "failed to open file:" << std::endl;
        }
        std::cout << line << std::endl;
        std::vector<std::string> elems;
        elems = split(line, ' ');

        std::vector< std::pair<std::string, float> > fv;
        std::string label = elems[0];
        for (int i=1; i < elems.size(); ++i) {

            std::vector<std::string> f_v = split(elems[i], ':');

            std::string f = f_v[0];
            float v = atof(f_v[1].c_str());
            std::cout << f << " " << v << std::endl;

            fv.push_back(std::make_pair(f, v));
        }

        y.push_back(label);
        x.push_back(fv);
    }

    Learner* p = new Perceptron();

    for (int t=0; t<3; t++) {
        p->fit2(x, y);
    }

    p->save("model");
    return 0;
}
