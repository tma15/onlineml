#include <string>
#include <vector>

#include <onlineml/common/dict.hpp>

void split(const std::string &s, char delim, std::vector<std::string> &elems); 
std::vector<std::string> split(const std::string &s, char delim); 

void read_data(const char* filename,
    Dict& labels,
    Dict& features,
    std::vector<size_t>& y,
    std::vector< std::vector< std::pair<size_t, float> > >& x);
