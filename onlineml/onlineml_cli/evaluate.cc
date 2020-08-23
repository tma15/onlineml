#include "../onlineml.h"

int main(int argc, char** argv) {
  onlineml::ArgumentParser parser;
  parser.add_argument("--input_file");
  parser.add_argument("--model_dir");
  parser.add_argument("--verbose");
  parser.parse_args(argc, argv);

  std::string input_file = parser.get<std::string>("input_file");
  std::string model_dir = parser.get<std::string>("model_dir");
  int verbose = parser.get<int>("verbose");

  onlineml::Dictionary feature_dictionary = onlineml::Dictionary::load(model_dir + "/feature.dict");
  onlineml::Dictionary label_dictionary = onlineml::Dictionary::load(model_dir + "/label.dict");

  feature_dictionary.freeze(true);
  label_dictionary.freeze(true);

  std::unique_ptr<onlineml::OnlineMLClassifier> classifier = std::unique_ptr<onlineml::OnlineMLClassifier>(
      new onlineml::Perceptron(
        feature_dictionary.size(),
        label_dictionary.size()));
  classifier->load(model_dir + "/classifier.om");

  unsigned num_correct = 0;
  unsigned num_total = 0;

  onlineml::DataLoader data_loader(
    input_file,
    feature_dictionary,
    label_dictionary);
  for (auto &labeled_vec : data_loader) {
    unsigned argmax = classifier->predict(labeled_vec.vector());

    if (verbose > 0) {
      std::cout << label_dictionary.get_elem(labeled_vec.label())
                << " " << label_dictionary.get_elem(argmax)
                << std::endl;
    }

    if (argmax == labeled_vec.label()) {
      num_correct++;
    }
    num_total++;
  }
  float accuracy = 100. * (float)num_correct / (float)num_total;
  std::cout << accuracy << " (" << num_correct << " / " << num_total << ")" << std::endl;

  return 0;
}
