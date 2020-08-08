#include "perceptron.h"


namespace onlineml {

float Perceptron::fit(const LabeledSparseVector &data) {
  unsigned argmax = predict(data.vector());
  if (argmax != data.label()) {
    for (unsigned i = 0; i < data.vector().size(); ++i) {
      unsigned feat_id = data.vector().at(i).id;
      weight_.index_add_({argmax, feat_id}, -1.);
      weight_.index_add_({data.label(), feat_id}, 1.);
    }
    return 1.;
  }
  return 0.;
}



unsigned Perceptron::predict(const SparseVector &data) {
  Tensor output = weight_ * data;
  unsigned argmax = output.argmax();
  return argmax;
}



void Perceptron::save(const std::string &filename) {
  ::onlineml::save(weight_, filename);
}



void Perceptron::load(const std::string &filename) {
  ::onlineml::load(weight_, filename);
}

} // namespace onlineml
