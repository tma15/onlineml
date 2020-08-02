#include "../onlineml.h"

int main(int argc, char** argv) {

  onlineml::Dictionary feature_dictionary = onlineml::Dictionary::load("./feature.dict");
  onlineml::Dictionary label_dictionary = onlineml::Dictionary::load("./label.dict");

  feature_dictionary.freeze(true);
  label_dictionary.freeze(true);

  std::unique_ptr<onlineml::OnlineMLClassifier> classifier = std::unique_ptr<onlineml::OnlineMLClassifier>(
      new onlineml::Perceptron(
        feature_dictionary.size(),
        label_dictionary.size()));
  classifier->load("classifier.om");

  unsigned num_correct = 0;
  unsigned num_total = 0;

  onlineml::DataLoader data_loader(
    argv[1],
    feature_dictionary,
    label_dictionary);
  for (auto &labeled_vec : data_loader) {
    unsigned argmax = classifier->predict(labeled_vec.vector());
    if (argmax == labeled_vec.label()) {
      num_correct++;
    }
    num_total++;
  }
  float accuracy = 100. * (float)num_correct / (float)num_total;
  std::cout << accuracy << " (" << num_correct << " / " << num_total << ")" << std::endl;

  return 0;
}
