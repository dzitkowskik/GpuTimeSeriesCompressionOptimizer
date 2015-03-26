/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

#include "core/logger.h"
#include "core/config.h"
#include <signal.h>
#include "store/store.h"

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

void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("logger.prop"));
}

int main(int argc, char* argv[])
{
  ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
  initialize_logger();

  // START THE PROGRAM HERE

  return wait_to_terminate();
}
