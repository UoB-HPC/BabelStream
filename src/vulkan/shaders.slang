// Copyright (c) 2026 Stream HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#if USE_DOUBLE
typedef double StreamType;
#else
typedef float StreamType;
#endif

[vk::constant_id(0)] const uint WORKGROUP_SIZE = 128;
[vk::constant_id(1)] const uint ELEMENT_COUNT = 0;
[vk::constant_id(2)] const uint DOT_GRID_SIZE = 0;
[vk::constant_id(3)] const StreamType INIT_A = 0;
[vk::constant_id(4)] const StreamType INIT_B = 0;
[vk::constant_id(5)] const StreamType INIT_C = 0;
[vk::constant_id(6)] const StreamType SCALAR = 0;

[[vk::binding(0, 0)]] RWStructuredBuffer<StreamType> gA;
[[vk::binding(1, 0)]] RWStructuredBuffer<StreamType> gB;
[[vk::binding(2, 0)]] RWStructuredBuffer<StreamType> gC;
[[vk::binding(3, 0)]] RWStructuredBuffer<StreamType> gPartial;

groupshared StreamType scratch[WORKGROUP_SIZE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void initShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gA[tid.x] = INIT_A;
    gB[tid.x] = INIT_B;
    gC[tid.x] = INIT_C;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void copyShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gC[tid.x] = gA[tid.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void mulShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gB[tid.x] = SCALAR * gC[tid.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void addShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gC[tid.x] = gA[tid.x] + gB[tid.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void triadShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gA[tid.x] = gB[tid.x] + SCALAR * gC[tid.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void nstreamShader(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= ELEMENT_COUNT) return;
    gA[tid.x] += gB[tid.x] + SCALAR * gC[tid.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void dotShader(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID)
{
    StreamType local = 0;

    for (uint i = dispatchThreadID.x; i < ELEMENT_COUNT; i += DOT_GRID_SIZE)
    {
        local += gA[i] * gB[i];
    }

    scratch[groupThreadID.x] = local;

    for (uint offset = WORKGROUP_SIZE >> 1; offset > 0; offset >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (groupThreadID.x < offset)
        {
            scratch[groupThreadID.x] += scratch[groupThreadID.x + offset];
        }
    }

    if (groupThreadID.x == 0)
    {
        gPartial[groupID.x] = scratch[0];
    }
}
