#include "classifiers/perceptron.h"
#include "data.h"
#include "trainer.h"

namespace onlineml {


OnlineMLTrainer::OnlineMLTrainer(OnlineMLTrainerOptions options) : options_(options) {
  data_loader_ = DataLoader(options_.input_file());
}

int OnlineMLTrainer::train() {

  std::unique_ptr<OnlineMLClassifier> classifier = std::unique_ptr<OnlineMLClassifier>(
      new Perceptron(
        data_loader_.feature_dictionary().size(),
        data_loader_.label_dictionary().size()));

  unsigned max_epoch = 20;
  for (unsigned epoch = 0; epoch < max_epoch; ++epoch) {
    unsigned num_total = 0;
    float num_wrong = 0;
    for (auto &labeled_vec : data_loader_) {
      ++num_total;
      num_wrong += classifier->fit(labeled_vec);
    }

    float num_correct = num_total - num_wrong;
    float accuracy = 100. * (float)num_correct / (float)num_total;
    std::cout << "epoch:" << epoch + 1;
    std::cout << " acc:" << accuracy;
    std::cout << " (" << num_correct << "/" << num_total << ")" << std::endl;
  }
  classifier->save("classifier.om");
  data_loader_.feature_dictionary().save("feature.dict");
  data_loader_.label_dictionary().save("label.dict");
  return 0;
}

} // namespace onlineml
