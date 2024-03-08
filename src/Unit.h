#pragma once
#include <iostream>

enum { MegaByte, GigaByte, MibiByte, GibiByte };

// Units for output:
struct Unit {
  int value;
  Unit(int v) : value(v) {}
  double fmt(double bytes) {
    switch(value) {
    case MibiByte: return pow(2.0, -20.0) * bytes;
    case MegaByte: return 1.0E-6 * bytes;
    case GibiByte: return pow(2.0, -30.0) * bytes;
    case GigaByte: return 1.0E-9 * bytes;
    default: std::cerr << "Unimplemented!" << std::endl; abort();
    }
  }
  char const* str() {
    switch(value) {
    case MibiByte: return "MiB";
    case MegaByte: return "MB";
    case GibiByte: return "GiB";
    case GigaByte: return "GB";
    default: std::cerr << "Unimplemented!" << std::endl; abort();
    }
  }
  Unit kibi() {
    switch(value) {
    case MegaByte: return Unit(MibiByte);
    case GigaByte: return Unit(GibiByte);
    default: return *this;
    }
  }
  Unit byte() {
    switch(value) {
    case MibiByte: return Unit(MegaByte);
    case GibiByte: return Unit(GigaByte);
    default: return *this;
    }
  }
  char const* lower() {
    switch(value) {
    case MibiByte: return "mibytes";
    case MegaByte: return "mbytes";
    case GibiByte: return "gibytes";
    case GigaByte: return "gbytes";
    default: std::cerr << "Unimplemented!" << std::endl; abort();
    }
  }
};
