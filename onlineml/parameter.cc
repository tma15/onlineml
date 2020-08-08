#include <stdio.h>
#include "parameter.h"


namespace onlineml {


Tensor::Tensor(std::initializer_list<unsigned> dims) : dims_(dims.begin(), dims.end()) {
  numel_ = 1;
  for (auto d : dims_) {
    numel_ *= d;
  }
  
  // TODO: make intrusive_ptr
  data_ = std::shared_ptr<float>(new float[numel_]);
  for (unsigned i = 0; i < numel_; ++i) {
    data_.get()[i] = 0.;
  }
//  storage_ = std::make_shared<Storage>();
//  storage_->allocate(numel_);

  strides_ = std::vector<unsigned>(dims_.size());
  for (unsigned i = 0; i < dims_.size(); ++i) {
    unsigned stride = 1;
    for (unsigned j = i + 1; j < dims_.size(); ++j) {
      stride *= dims_[j];
    }
    strides_[i] = stride;
  }

}


Tensor::Tensor(std::vector<unsigned> dims) : dims_(dims) {
  numel_ = 1;
  for (auto d : dims_) {
    numel_ *= d;
  }
  data_ = std::make_shared<float>(numel_);
}


Tensor::Tensor(std::shared_ptr<float> data, std::vector<unsigned> dims) : dims_(dims) {
  numel_ = 1;
  for (auto d : dims_) {
    numel_ *= d;
  }
  data_ = data;
}



const unsigned Tensor::argmax() const {
  unsigned argmax_ = 0;
  float max_ = data_.get()[0];

  for (unsigned i = 1; i < numel_; ++i) {
    if (data_.get()[i] > max_) {
      argmax_ = i;
      max_ = data_.get()[i];
    }
  }
  return argmax_;
}



Tensor Tensor::operator[](unsigned idx) {
  unsigned numel_i = numel_ / dims_[0];
  unsigned offset = idx * numel_i;
  std::shared_ptr<float> data_i(data_, data_.get() + offset);
  std::vector<unsigned> dims_i(dims_.begin() + 1, dims_.end());
  return Tensor(data_i, dims_i);
}



Tensor Tensor::reshape(std::initializer_list<unsigned> dims) {
  Tensor tensor(dims);
  tensor.data_ptr(data_);
  return tensor;
}



void Tensor::index_add_(std::initializer_list<unsigned> dims, float val) {
  unsigned offset = 0;
  std::vector<unsigned> tmp(dims.begin(), dims.end());
  for (unsigned d = 0; d < tmp.size(); ++d) {
    offset += tmp[d] * strides_[d];
  }
  data_.get()[offset] += val;
}


Tensor arange(unsigned numel, std::initializer_list<unsigned> dims) {
  Tensor tensor(dims);
  float *data = tensor.data_ptr().get();
  for (unsigned i = 0; i < numel; ++i) {
    data[i] = (float)i;
  }
  return tensor;
}


Tensor zeros(std::initializer_list<unsigned> dims) {
  return Tensor(dims);
}



Tensor operator*(Tensor tensor, const SparseVector &vec) {
  Tensor out = zeros({tensor.dim()[0]});

  for (unsigned y = 0; y < tensor.dim()[0]; ++y) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      unsigned key = vec.at(i).id;
      float value = vec.at(i).value;
      out.index_add_({y}, value * tensor[y][key].scalar());
    }
  }
  return out;
}



void save(Tensor tensor, const std::string &file) {
  FILE *fp = fopen(file.c_str(), "wb");
  fwrite(tensor.data_ptr().get(), sizeof(float), tensor.numel(), fp);
  fclose(fp);
}



void load(Tensor tensor, const std::string &file) {
  FILE *fp = fopen(file.c_str(), "rb");
  std::shared_ptr<float> data = std::shared_ptr<float>(new float[tensor.numel()]);
  fread(data.get(), sizeof(float), tensor.numel(), fp);
  memcpy(tensor.data_ptr().get(), data.get(), sizeof(float) * tensor.numel());;
  fclose(fp);
}

} // namespace onlineml
