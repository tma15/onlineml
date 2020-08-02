#pragma once
#include "../classifier.h"
#include "../parameter.h"


namespace onlineml {

class Perceptron : public OnlineMLClassifier {
  public:
    Perceptron() {}

    Perceptron(unsigned input_size, unsigned output_size)
    : input_size_(input_size), output_size_(output_size) {

      weight_ = Tensor({output_size, input_size});
    }

    float fit(const LabeledSparseVector &data);
    unsigned predict(const SparseVector &vec);
    void save(const std::string &file_name);
    void load(const std::string &file_name);

  private:
    unsigned input_size_;
    unsigned output_size_;
    Tensor weight_;
};
 
} // namespace onlineml
