#pragma once
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_map>

namespace onlineml {


class ArgumentParser {
  public:
    ArgumentParser() {}
    void add_argument(const std::string &option);
    void add_argument(const std::string &option, const std::string &defval);
    void parse_args(int argc, char** argv);
    void print_help();

    template<typename T> 
    T get(const std::string &option);

  private:
    std::string get_key(const std::string &option);
    std::unordered_map<std::string, std::string> args_;
    std::string command_;
};


void ArgumentParser::add_argument(const std::string &option) {
  args_[get_key(option)] = "";
}

void ArgumentParser::add_argument(const std::string &option, const std::string &defval) {
  args_[get_key(option)] = defval;
}

void ArgumentParser::parse_args(int argc, char** argv) {
  command_ = std::string(argv[0]);

  for (unsigned i = 0; i < argc; ++i) {
    std::string key = get_key(std::string(argv[i]));


    if (args_.find(key) != args_.end()) {
      std::string value = std::string(argv[i + 1]);
      args_[key] = value;
      i++;
    }

    if (key == "h" || key == "help") {
      print_help();
      exit(0);
    }

  }
}

template<typename T>
T ArgumentParser::get(const std::string &option) {
  T ret;

  std::string value = args_[option];
  std::stringstream ss(value);
  ss >> ret;
  return ret;
}


void ArgumentParser::print_help() {
  std::cout << "Usage of " << command_ << ":" << std::endl;
  for (auto it = args_.begin(); it != args_.end(); it++) {
    std::string hyphen;
    if (it->first.size() == 1) {
      hyphen = "-";
    } else {
      hyphen = "--";
    }
    std::cout << "\t" << hyphen << it->first << std::endl;
  }
}


std::string ArgumentParser::get_key(const std::string &option) {
  std::string key;

  if (option[0] == '-' && option[1] == '-') {
    key = std::string(option.begin() + 2, option.end());
  } else if (option[0] == '-' && option[1] != '-') {
    key = std::string(option.begin() + 1, option.end());
  }
  return key;
}

} // namespace onlineml
