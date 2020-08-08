#include <fstream>

#include "data.h"
#include "string_process.h"


namespace onlineml {


bool Dictionary::has_elem(const std::string &elem) const {
  return id_.find(elem) != id_.end();
}

unsigned Dictionary::add_elem(const std::string &elem) {
  if (!has_elem(elem)) {
    unsigned idx = elem_.size();
    elem_.push_back(elem);
    id_[elem] = idx;
    return idx;
  } else {
  return id_[elem];
  }
}

unsigned Dictionary::get_idx(const std::string &elem) const {
  return id_.at(elem);
}

const std::string &Dictionary::get_elem(unsigned idx) const {
  return elem_[idx];
}



void Dictionary::save(const std::string &filename) const {
  std::ofstream ofs(filename);
  for (auto &elem : elem_) {
    ofs << elem << std::endl;
  }
}



Dictionary Dictionary::load(const std::string &filename) {
  Dictionary dict;
  std::ifstream ifs(filename);
  std::string line;
  while (getline(ifs, line)) {
    dict.add_elem(line);
  }
  return dict;
}



void SparseVector::push_back(unsigned id, float value) {
  features_.push_back(Feature(id, value));
}


std::ostream &operator<<(std::ostream &os, const Feature &feature) {
  os << "Feature(id:" << feature.id << " value:" << feature.value << ")";
  return os;
}



std::ostream &operator<<(std::ostream &os, const SparseVector &vec) {
  os << "SparseVector(" << std::endl;
  for (unsigned i = 0; i < vec.size(); ++i) {
    const Feature f = vec.at(i);
    os << "\t" << f;
    if (i < vec.size() - 1) {
      os << ",";
    }
    os << std::endl;
  }
  os << ")";
  return os;
}



void DataLoader::load(const std::string &input_file) {
  std::ifstream ifs(input_file);

  std::string line;
  while (getline(ifs, line)) {
    SparseVector vec;

    std::vector<std::string> elems;
    split(line, ' ', &elems);

    std::string label = elems[0];

    unsigned label_id;
    if (label_dictionary_.freezed()) {
      label_id = label_dictionary_.get_idx(label);
    } else {
      label_id = label_dictionary_.add_elem(label);
    }

    for (unsigned i = 1; i < elems.size(); ++i) {
      std::vector<std::string> fv_str;

      split(elems[i], ':', &fv_str);
      std::string feature = fv_str[0];
      float value = std::stof(fv_str[1]);

      unsigned feature_id;
      if (feature_dictionary_.freezed()) {
        if (feature_dictionary_.has_elem(feature)) {
          feature_id = feature_dictionary_.get_idx(feature);
          vec.push_back(feature_id, value);
        }
      } else {
        feature_id = feature_dictionary_.add_elem(feature);
        vec.push_back(feature_id, value);
      }
    }

    LabeledSparseVector labeled_vec(vec, label_id);
    data_.push_back(labeled_vec);
  }
}


DataLoader::DataLoader(const std::string &input_file) : input_file_(input_file) {
  load(input_file_);
}



DataLoader::DataLoader(const std::string &input_file,
                       const Dictionary &feature_dictionary,
                       const Dictionary &label_dictionary)
  : input_file_(input_file),
    feature_dictionary_(feature_dictionary),
    label_dictionary_(label_dictionary) {

  load(input_file_);
}

} // namespace onlineml
