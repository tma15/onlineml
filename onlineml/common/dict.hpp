#include <vector>
//#include <map>
#include <unordered_map>
#include <string.h>
#include <iostream>

#ifndef INCLUDED_DICT
#define INCLUDED_DICT
class Dict {
    public:
        std::vector<std::string> elems;
//        std::map<std::string, size_t> ids;
        std::unordered_map<std::string, size_t> ids;

        Dict(){};
        size_t size() { return this->elems.size();};
        bool has_elem(std::string elem);
        size_t add_elem(std::string elem);
        size_t get_id(std::string elem);
        std::string get_elem(size_t id);
};

bool Dict::has_elem(std::string elem) {
    if (this->ids.count(elem)==1) {
        return true;
    }
    return false;
}

size_t Dict::add_elem(std::string elem) {
    size_t id = this->elems.size();
    this->elems.push_back(elem);
    this->ids[elem] = id;
    return id;
}

std::string Dict::get_elem(size_t id) {
    if (this->elems.size() <= id) {
        return "";
    }
    return this->elems[id];
}

size_t Dict::get_id(std::string elem) {
    if (this->ids.count(elem) == 0) {
        return -1;
    }
    return this->ids[elem];
}
#endif
