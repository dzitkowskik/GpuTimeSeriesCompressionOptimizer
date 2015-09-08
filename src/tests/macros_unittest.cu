#include "core/cuda_macros.cuh"
#include <gtest/gtest.h>

TEST(BITLEN, Macros_Alt_Bitlen_Small)
{
	int value = 12345;
	int result = BITLEN(value);
	int expected = 14;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Alt_Bitlen_Big)
{
	int value = 2147268225;
	int result = BITLEN(value);
	int expected = 31;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Alt_Bitlen_Negative_Small)
{
	int value = -12345;
	int result = BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(BITLEN, Macros_Alt_Bitlen_Negative_Big)
{
	int value = -2147268227;
	int result = BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Small)
{
	int value = 12345;
	int result = ALT_BITLEN(value);
	int expected = 14;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Big)
{
	int value = 2147268225;
	int result = ALT_BITLEN(value);
	int expected = 31;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Negative_Small)
{
	int value = -12345;
	int result = ALT_BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Negative_Big)
{
	int value = -2147268227;
	int result = ALT_BITLEN(value);
	int expected = 32;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_Zero)
{
	int value = 0;
	int result = ALT_BITLEN(value);
	int expected = 1;
	EXPECT_EQ(expected, result);
}

TEST(ALT_BITLEN, Macros_Alt_Bitlen_One)
{
	int value = 1;
	int result = ALT_BITLEN(value);
	int expected = 1;
	EXPECT_EQ(expected, result);
}
