#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>

namespace onlineml {


class Dictionary {
  public:
    Dictionary() : freezed_(false) {}

    const unsigned size() const { return elem_.size(); }

    bool has_elem(const std::string &elem) const;
    unsigned add_elem(const std::string &elem);
    unsigned get_idx(const std::string &elem) const;
    const std::string &get_elem(unsigned idx) const;

    void freeze(bool freeze) { freezed_ = freeze;; }
    const bool freezed() const { return freezed_; }

    void save(const std::string &filename) const;
    static Dictionary load(const std::string &filename);

  private:
    std::vector<std::string> elem_;
    std::unordered_map<std::string, unsigned> id_;
    bool freezed_;
};


class Feature {
  public:
    Feature(unsigned id, float value) : id(id), value(value) {}

    friend std::ostream &operator<<(std::ostream &os, const Feature &feature);

    unsigned id;
    float value;
};


class SparseVector {
  public:
    SparseVector() {}
    SparseVector(const std::vector<Feature> &features) : features_(features) {}

    void push_back(unsigned id, float value);

    const unsigned size() const { return features_.size(); }

    const Feature &at(unsigned id) const { return features_[id]; }

    friend std::ostream &operator<<(std::ostream &os, const SparseVector &vec);

  private:
    std::vector<Feature> features_;
};


class LabeledSparseVector : public SparseVector {
  public:
    LabeledSparseVector(
        const SparseVector &vec,
        unsigned label)
      : vec_(vec), label_(label) {}

    const SparseVector &vector() const { return vec_; }
    const unsigned &label() const { return label_; }

  private:
    SparseVector vec_;
    unsigned label_;
};



class DataLoader {
  public:
    typedef std::vector<LabeledSparseVector>::iterator iterator;
    typedef std::vector<LabeledSparseVector>::const_iterator const_iterator;

    DataLoader() {};

    DataLoader(const std::string &input_file);

    DataLoader(const std::string &input_file,
               const Dictionary &feature_dictionary,
               const Dictionary &label_dictionary);

    void load(const std::string &input_file);

    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator end() const { return data_.end(); }

    const Dictionary &feature_dictionary() { return feature_dictionary_; }
    const Dictionary &label_dictionary() { return label_dictionary_; }

  private:
    std::string input_file_;

    Dictionary label_dictionary_;
    Dictionary feature_dictionary_;

    std::vector<LabeledSparseVector> data_;
};
  
} // namespace onlineml
