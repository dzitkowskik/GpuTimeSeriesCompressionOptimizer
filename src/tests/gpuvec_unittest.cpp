#include "gpuvec_unittest.h"
#include <thrust/version.h>

namespace ddj {

// check thrust version
TEST_F(GpuVecTest, ThrustVersion)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    RecordProperty("Thrust version major", major);
    RecordProperty("Thrust version minor", minor);
    EXPECT_EQ(1, major);
    EXPECT_EQ(8, minor);
}

} /* namespace ddj */
