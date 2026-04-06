# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-src"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-build"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/tmp"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/src"
  "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/soham/fin_auditor/hf auditor/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
