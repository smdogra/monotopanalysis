#include "../interface/FileMerger.h"

#include <iostream>

int
main(int _argc, char** _argv)
{
  if (_argc < 3) {
    std::cerr << "Usage: padd OUTPUT INPUT [..]" << std::endl;
    return 1;
  }

  panda::FileMerger merger;
  for (int iA(2); iA != _argc; ++iA)
    merger.addInput(_argv[iA]);

  merger.merge(_argv[1]);

  return 0;
}
