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

    OnlineMLTrainerOptions max_epoch(unsigned max_epoch) {
      max_epoch_ = max_epoch;
      return *this;
    }

    const std::string &input_file() { return input_file_; }
    const unsigned &max_epoch() { return max_epoch_; }

  private:
    std::string input_file_;
    unsigned max_epoch_;
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
