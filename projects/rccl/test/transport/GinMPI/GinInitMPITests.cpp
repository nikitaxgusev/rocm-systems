/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "gin.h"

#include <cstdlib>
#include <cstring>

#ifdef MPI_TESTS_ENABLED

extern bool rcclUseAinic();

using namespace MPITestConstants;

class GinInitMPITest : public MPITestBase
{
protected:
    ncclGin_t* AssignedGin()
    {
        auto* comm = reinterpret_cast<struct ncclComm*>(getActiveCommunicator());
        return comm->sharedRes->ginState.ncclGin;
    }
};

TEST_F(GinInitMPITest, GinEnableZeroSkipsInit)
{
    const char* e = std::getenv("NCCL_GIN_ENABLE");
    if (e == nullptr || std::strcmp(e, "0") != 0)
        GTEST_SKIP() << "Requires NCCL_GIN_ENABLE=0";

    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    EXPECT_EQ(AssignedGin(), nullptr) << "GIN must be skipped when NCCL_GIN_ENABLE=0";
}

TEST_F(GinInitMPITest, AinicSelectsCastBackend)
{
    if (!rcclUseAinic())
        GTEST_SKIP() << "Requires AINIC hardware";

    const char* e = std::getenv("NCCL_GIN_ENABLE");
    if (e != nullptr && std::strcmp(e, "0") == 0)
        GTEST_SKIP() << "GIN disabled by NCCL_GIN_ENABLE=0";

    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    ncclGin_t* gin = AssignedGin();
    if (gin == nullptr)
        GTEST_SKIP() << "No GIN-capable devices enabled on this host";

    // The generic net_ib proxy (ncclGinIbProxy) and the ionic net_ib_cast
    // backends all report the name "GIN_IB_PROXY", so the backend identity must
    // be checked through its function table rather than its name. The ionic
    // cast entry point (IbCastGinIb) morphs into IbCastGinIbProxy or
    // IbCastGinIbGdaki during init(); both cast variants share
    // IbCastGinIbFinalize, which differs from the generic backend's finalize.
    TEST_INFO("Assigned GIN backend: %s (finalize=%p)", gin->name, (void*)gin->finalize);
    EXPECT_NE(gin->finalize, ncclGinIbProxy.finalize)
        << "AINIC must not use the generic IB GIN backend";
    EXPECT_EQ(gin->finalize, IbCastGinIbProxy.finalize)
        << "AINIC must use the ionic ib-cast GIN backend";
}

#endif // MPI_TESTS_ENABLED
