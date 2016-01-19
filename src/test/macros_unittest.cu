#include "core/cuda_macros.cuh"
#include <cuda.h>
#include <gtest/gtest.h>

TEST(BITLEN, Macros_Bitlen_Int_Zero)
{
	unsigned int min = 0;
	unsigned int max = 0;
	int result = BITLEN(min, max);
	int expected = 1;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Int_One)
{
	unsigned int min = 0;
	unsigned int max = 1;
	int result = BITLEN(min, max);
	int expected = 1;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Int_Small)
{
	unsigned int min = 0;
	unsigned int max = 12345;
	int result = BITLEN(min, max);
	int expected = 14;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Int_Big)
{
	int min = 0;
	int max = 2147268225;
	int result = BITLEN(min, max);
	int expected = 31;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Int_Negative_Small)
{
	int min = -12345;
	int max = -12345;
	int result = BITLEN(min, max);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Int_Negative_Big)
{
	int min = -2147268227;
	int max = 147268227;
	int result = BITLEN(min, max);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Short_Small)
{
	short min = 0;
	short max = 12345;
	int result = BITLEN(min, max);
	int expected = 14;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Char_Small)
{
	char min = 0;
	char max = 45;
	int result = BITLEN(min, max);
	int expected = 6;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Bitlen_Long_Big)
{
	long min = 0;
	long max = 22147268225;
	int result = BITLEN(min, max);
	int expected = 35;
	EXPECT_EQ(expected, result);
}



//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////



TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_Small)
{
	int value = 12345;
	int result = ALT_BITLEN(value);
	int expected = 14;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_Big)
{
	int value = 2147268225;
	int result = ALT_BITLEN(value);
	int expected = 31;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_Negative_Small)
{
	int value = -12345;
	int result = ALT_BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_Negative_Big)
{
	int value = -2147268227;
	int result = ALT_BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_Zero)
{
	int value = 0;
	int result = ALT_BITLEN(value);
	int expected = 1;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Int_One)
{
	int value = 1;
	int result = ALT_BITLEN(value);
	int expected = 1;
	EXPECT_EQ(expected, result);
}
