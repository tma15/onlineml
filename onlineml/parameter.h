#pragma once
#include <memory>

#include "data.h"

namespace onlineml {


class Storage {
  public:
    Storage() {}
};


class Tensor {
  public:
    Tensor() : numel_(0) {}; 
    Tensor(std::initializer_list<unsigned> dims); 
    Tensor(std::vector<unsigned> dims); 
    Tensor(std::shared_ptr<float> data, std::vector<unsigned> dims); 

    std::shared_ptr<float> data_ptr() { return data_; }

    void data_ptr(std::shared_ptr<float> data) { data_ = data; }

    const unsigned numel() const { return numel_; }
    const std::vector<unsigned> dim() const { return dims_; }
    const unsigned argmax() const;

    Tensor operator[](unsigned idx);

    float scalar() { return data_.get()[0]; }

    Tensor reshape(std::initializer_list<unsigned> dims);
    void index_add_(std::initializer_list<unsigned> dims, float val);

  private:
    std::shared_ptr<float> data_;
    std::vector<unsigned> dims_;
    std::vector<unsigned> strides_;
    unsigned numel_;
};


Tensor zeros(std::initializer_list<unsigned> dims);
Tensor arange(unsigned numel, std::initializer_list<unsigned> dims);
Tensor reshape(std::initializer_list<unsigned> dims);

Tensor operator*(Tensor tensor, const SparseVector &vec);

void save(Tensor tensor, const std::string &file);
void load(Tensor tensor, const std::string &file);


} // namespace onlineml
