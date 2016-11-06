#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <onlineml/common/dict.hpp>

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void read_data(const char* filename,
    Dict& labels,
    Dict& features,
    std::vector<size_t>& y,
    std::vector< std::vector< std::pair<size_t, float> > >& x) {

    std::ifstream ifs(filename);
    std::string line;

//    std::vector<size_t> y;
//    std::vector< std::vector< std::pair<size_t, float> > > x;

//    Dict features;
//    Dict labels;
    clock_t start_data = clock();

    if (ifs.fail()) {
        std::cerr << "failed to open file:" << std::endl;
    }

    while(getline(ifs, line)) {
        std::vector<std::string> elems;
        split(line, ' ', elems);

        std::string label = elems[0];
        
        std::vector< std::pair<size_t, float> > fv;
        size_t num_elem = elems.size();
        fv = std::vector< std::pair<size_t, float> >(num_elem-1);
        for (size_t i=1; i < num_elem; ++i) {

            std::vector<std::string> f_v;
            split(elems[i], ':', f_v);

            std::string f = f_v[0];
            float v = atof(f_v[1].c_str());

            size_t fid;
            if (!features.has_elem(f)) {
                fid = features.add_elem(f);
            } else {
                fid = features.get_id(f);
            }
            std::pair<size_t, float> ftval = std::make_pair(fid, v);
            fv[i-1] = ftval;
        }

        size_t yid;
        if (!labels.has_elem(label)) {
            yid = labels.add_elem(label);
        } else {
            yid = labels.get_id(label);
        }

        y.push_back(yid);
        x.push_back(fv);
    }
}
