/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

#include "logger.h"
#include "config.h"
#include <signal.h>
#include "store.h"
#include "thrust_test.cuh"
#include <gtest/gtest.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("logger.prop"));
}

int wait_to_terminate()
{
  //wait for SIGINT
  int sig_number;
  sigset_t signal_set;\
  sigemptyset (&signal_set);
  sigaddset (&signal_set, SIGINT);
  sigwait (&signal_set, &sig_number);
  return EXIT_SUCCESS;
}

int configure_tests(int argc, char* argv[])
{
  auto conf = ddj::Config::GetInstance();

  bool enableTest = false;
  if(conf->HasValue("test"))
  {
    ::testing::GTEST_FLAG(filter) = "*Test*";
    enableTest = true;
  }
  else if(conf->HasValue("performance"))
  {
    ::testing::GTEST_FLAG(filter) = "*Performance*";
    enableTest = true;
  }
  if(enableTest)
  {
    Logger::getRoot().removeAllAppenders();
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_repeat = 1;
    return 1;
  }
  return 0;
}

int main(int argc, char* argv[])
{
  ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
  initialize_logger();
  if(configure_tests(argc, argv)) return RUN_ALL_TESTS();

  // START THE PROGRAM HERE
  run_cuda();

  return wait_to_terminate();
}
