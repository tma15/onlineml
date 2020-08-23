#include <iostream>

#include "../onlineml.h"


int main(int argc, char** argv) {
  onlineml::ArgumentParser parser;
  parser.add_argument("--input_file");
  parser.add_argument("--max_epoch", "10");
  parser.parse_args(argc, argv);

  onlineml::OnlineMLTrainerOptions options;
  options = options.input_file(parser.get<std::string>("input_file"));
  options = options.max_epoch(parser.get<unsigned>("max_epoch"));

  std::cout << "input_file: " << options.input_file() << std::endl;
  std::cout << "max_epoch: " << options.max_epoch() << std::endl;

  onlineml::OnlineMLTrainer trainer(options);
  trainer.train();
  return 0;
}
