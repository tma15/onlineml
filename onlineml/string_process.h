#pragma once

#include <string>
#include <vector>

namespace onlineml {

inline void split(std::string const& original, char separator, std::vector<std::string>* elems ) {
  std::string::const_iterator start = original.begin();
  std::string::const_iterator end = original.end();
  std::string::const_iterator next = std::find(start, end, separator);

    while (next != end) {
      elems->push_back(std::string(start, next));
      start = next + 1;
      next = std::find(start, end, separator);
    }
    elems->push_back(std::string(start, next));
}

} // namespace onlineml
