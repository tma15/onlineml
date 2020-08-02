#pragma once
#include <iostream>

#include "data.h"


namespace onlineml {


class OnlineMLTrainerOptions {
  public:
    OnlineMLTrainerOptions input_file(const std::string &input_file) {
      input_file_ = input_file;
      return *this;
    }

    const std::string &input_file() { return input_file_; }

  private:
    std::string input_file_;
};


class OnlineMLTrainer {
  public:
    OnlineMLTrainer() {}

    OnlineMLTrainer(OnlineMLTrainerOptions options);

    int train();

  private:
    OnlineMLTrainerOptions options_;
    DataLoader data_loader_;
};

} // namespace onlineml
