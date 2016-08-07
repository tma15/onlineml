#include <vector>
#include <map>
#include <string.h>
#include <iostream>

#ifndef INCLUDED_DICT
#define INCLUDED_DICT
class Dict {
    public:
        std::vector<std::string> elems;
        std::map<std::string,  int> ids;

        Dict(){};
        int size() { return this->elems.size();};
        bool has_elem(std::string elem);
        void add_elem(std::string elem);
        int get_id(std::string elem);
        std::string get_elem(int id);
};

bool Dict::has_elem(std::string elem) {
    for (size_t i=0; i < this->elems.size(); i++) {
        std::string e = this->elems[i];
        if (elem == e) {
            return true;
        }
    }
    return false;
}

void Dict::add_elem(std::string elem) {
    int id = this->elems.size();
    this->elems.push_back(elem);
    this->ids.insert(std::map<std::string, int>::value_type(elem, id));
}

std::string Dict::get_elem(int id) {
    if (this->elems.size() <= id) {
        return "";
    }
    return this->elems[id];
}

int Dict::get_id(std::string elem) {
    if (this->ids.count(elem) == 0) {
        return -1;
    }
    return this->ids[elem];
}
#endif
