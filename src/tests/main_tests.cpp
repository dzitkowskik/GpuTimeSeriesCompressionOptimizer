#include <gtest/gtest.h>
#include "../logger.h"
#include "../config.h"

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
    Logger::getRoot().removeAllAppenders();
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_repeat = 1;
    return RUN_ALL_TESTS();
}
