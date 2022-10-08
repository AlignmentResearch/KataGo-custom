#include "../core/makedir.h"

#include <filesystem>

namespace fs = std::filesystem;

//------------------------
#include "../core/using.h"
//------------------------

// Equivalent to mkdir -p in bash
// Now just a wrapper around std::filesystem::create_directories as of C++17
void MakeDir::make(const string& path) { fs::create_directories(path); }
