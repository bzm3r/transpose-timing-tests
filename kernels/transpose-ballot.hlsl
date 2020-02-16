// Conversion of transpose-ballot.comp into SPIR-V, and then into HLSL using SPIRV-Cross.
static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

RWByteAddressBuffer _21 : register(u0, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint tix = gl_GlobalInvocationID.x;
    uint input_mask = _21.Load(tix * 4 + 0);
    for (uint i = 0u; i < 32u; i++)
    {
        uint shift_mask = uint(1 << int(i));
        uint4 vote = WaveActiveBallot((input_mask & shift_mask) != 0u);
        if (i == tix)
        {
            _21.Store(tix * 4 + 0, vote.x);
        }
    }
}

[numthreads(32, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
