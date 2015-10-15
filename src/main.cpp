/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

#include "core/logger.h"
#include "core/config.hpp"
#include <signal.h>
#include "tree/compression_node.hpp"

#include "util/generator/cuda_array_generator.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "tests/encode_decode_unittest_helper.hpp"
#include <boost/bind.hpp>

using namespace std;
using namespace ddj;

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

void test_dict_encoding()
{
	CudaArrayGenerator generator;
	auto d_int_random_data = generator.GenerateRandomIntDeviceArray(10000, 100, 1000);
	DictEncoding encoder;
	auto result = EncodeDecodeUnittestHelper::TestSize2<int>(
	boost::bind(&DictEncoding::Encode<int>, encoder, _1),
	boost::bind(&DictEncoding::Decode<int>, encoder, _1),
	d_int_random_data);
	if (result) printf("\n\nOK\n\n");
	else printf("\n\nFAIL\n\n");
}

int main(int argc, char* argv[])
{
	ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
	initialize_logger();

	// START THE PROGRAM HERE

	//  return wait_to_terminate();
	return 0;
}
