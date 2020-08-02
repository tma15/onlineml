#pragma once
#include "data.h"

namespace onlineml {

class OnlineMLClassifier {
  public:
    virtual float fit(const LabeledSparseVector &data)=0;
    virtual unsigned predict(const SparseVector &data)=0;
    virtual void save(const std::string &file_name)=0;
    virtual void load(const std::string &file_name)=0;
};
  
} // namespace onlineml
