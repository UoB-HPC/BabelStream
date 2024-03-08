#pragma once
#include <iostream>

// Units for output:
struct Unit {
  enum class Kind { MegaByte, GigaByte, TeraByte, MibiByte, GibiByte, TebiByte };
  Kind value;
  explicit Unit(Kind v) : value(v) {}
  double fmt(double bytes) const {
    switch(value) {
    case Kind::MibiByte: return std::pow(2.0, -20.0) * bytes;
    case Kind::MegaByte: return 1.0E-6 * bytes;
    case Kind::GibiByte: return std::pow(2.0, -30.0) * bytes;
    case Kind::GigaByte: return 1.0E-9 * bytes;
    case Kind::TebiByte: return std::pow(2.0, -40.0) * bytes;
    case Kind::TeraByte: return 1.0E-12 * bytes;      
    default: std::cerr << "Unimplemented!" << std::endl; std::abort();
    }
  }
  char const* str() const {
    switch(value) {
    case Kind::MibiByte: return "MiB";
    case Kind::MegaByte: return "MB";
    case Kind::GibiByte: return "GiB";
    case Kind::GigaByte: return "GB";
    case Kind::TebiByte: return "TiB";
    case Kind::TeraByte: return "TB";
    default: std::cerr << "Unimplemented!" << std::endl; std::abort();
    }
  }
};
