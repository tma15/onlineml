#include <iostream>

#include "../onlineml.h"


int main(int argc, char** argv) {
  onlineml::OnlineMLTrainerOptions options;
  for (unsigned i = 0; i < argc; ++i) {
    std::cout << i << ":" << argv[i] << std::endl;
  }
  options = options.input_file(argv[1]);

  std::cout << "input_file: " << options.input_file() << std::endl;

  onlineml::OnlineMLTrainer trainer(options);
  trainer.train();
  return 0;
}
