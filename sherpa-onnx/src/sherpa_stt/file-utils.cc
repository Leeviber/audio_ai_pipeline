// sherpa-onnx/csrc/file-utils.cc
//

#include "file-utils.h"

#include <fstream>
#include <string>

#include "log.h"

namespace sherpa_onnx {

bool FileExists(const std::string &filename) {
  return std::ifstream(filename).good();
}

void AssertFileExists(const std::string &filename) {
  if (!FileExists(filename)) {
    SHERPA_ONNX_LOG(FATAL) << filename << " does not exist!";
  }
}

}  // namespace sherpa_onnx
