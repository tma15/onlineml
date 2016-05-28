#include <map>

#ifndef INCLUDED_LEARNER
#define INCLUDED_LEARNER

#ifndef INCLUDED_DICT
//#include "include/dict.h"
#include <onlineml/dict.h>
#endif

class Learner {
    public:
        Dict features;
        Dict labels;
        std::vector< std::vector<float> > weight;

        Learner(){
        };
        ~Learner(){};
        virtual void fit(std::vector< std::map<std::string, float> > x, std::vector<std::string> y){
        };
        virtual int predict(std::map<std::string, float > x){};
        virtual const char* id2label(int id){};
        virtual void save(const char*){};
};

#endif
