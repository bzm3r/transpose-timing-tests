// Conversion of transpose-threadgroup.comp into SPIR-V, and then into HLSL using SPIRV-Cross.

static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

RWByteAddressBuffer _62 : register(u0, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint tg_bm[32];

uint shuffle_round(uint dst_tid, uint a, uint b, inout uint m, uint s)
{
    uint c;
    if ((dst_tid & s) == 0u)
    {
        c = b << s;
    }
    else
    {
        m = ~m;
        c = b >> s;
    }
    return (a & m) | (c & (~m));
}

void comp_main()
{
    uint dst_tid = gl_GlobalInvocationID.x;
    if (dst_tid == 0u)
    {
        uint _67[32];
        [unroll]
        for (int _2ident = 0; _2ident < 32; _2ident++)
        {
            [unroll]
            for (int _3ident = 0; _3ident < 32; _3ident++)
            {
                _67[_2ident][_3ident] = _62.Load(_3ident * 4 + _2ident * 4 + 0);
            }
        }
        tg_bm[0] = _67[0];
        tg_bm[1] = _67[1];
        tg_bm[2] = _67[2];
        tg_bm[3] = _67[3];
        tg_bm[4] = _67[4];
        tg_bm[5] = _67[5];
        tg_bm[6] = _67[6];
        tg_bm[7] = _67[7];
        tg_bm[8] = _67[8];
        tg_bm[9] = _67[9];
        tg_bm[10] = _67[10];
        tg_bm[11] = _67[11];
        tg_bm[12] = _67[12];
        tg_bm[13] = _67[13];
        tg_bm[14] = _67[14];
        tg_bm[15] = _67[15];
        tg_bm[16] = _67[16];
        tg_bm[17] = _67[17];
        tg_bm[18] = _67[18];
        tg_bm[19] = _67[19];
        tg_bm[20] = _67[20];
        tg_bm[21] = _67[21];
        tg_bm[22] = _67[22];
        tg_bm[23] = _67[23];
        tg_bm[24] = _67[24];
        tg_bm[25] = _67[25];
        tg_bm[26] = _67[26];
        tg_bm[27] = _67[27];
        tg_bm[28] = _67[28];
        tg_bm[29] = _67[29];
        tg_bm[30] = _67[30];
        tg_bm[31] = _67[31];
    }
    GroupMemoryBarrierWithGroupSync();
    uint s = 16u;
    uint src_tid = dst_tid ^ s;
    uint param = dst_tid;
    uint param_1 = tg_bm[dst_tid];
    uint param_2 = tg_bm[src_tid];
    uint param_3 = 65535u;
    uint param_4 = s;
    uint _188 = shuffle_round(param, param_1, param_2, param_3, param_4);
    tg_bm[dst_tid] = _188;
    GroupMemoryBarrierWithGroupSync();
    s = 8u;
    src_tid = dst_tid ^ s;
    uint param_5 = dst_tid;
    uint param_6 = tg_bm[dst_tid];
    uint param_7 = tg_bm[src_tid];
    uint param_8 = 16711935u;
    uint param_9 = s;
    uint _209 = shuffle_round(param_5, param_6, param_7, param_8, param_9);
    tg_bm[dst_tid] = _209;
    GroupMemoryBarrierWithGroupSync();
    s = 4u;
    src_tid = dst_tid ^ s;
    uint param_10 = dst_tid;
    uint param_11 = tg_bm[dst_tid];
    uint param_12 = tg_bm[src_tid];
    uint param_13 = 252645135u;
    uint param_14 = s;
    uint _230 = shuffle_round(param_10, param_11, param_12, param_13, param_14);
    tg_bm[dst_tid] = _230;
    GroupMemoryBarrierWithGroupSync();
    s = 2u;
    src_tid = dst_tid ^ s;
    uint param_15 = dst_tid;
    uint param_16 = tg_bm[dst_tid];
    uint param_17 = tg_bm[src_tid];
    uint param_18 = 858993459u;
    uint param_19 = s;
    uint _250 = shuffle_round(param_15, param_16, param_17, param_18, param_19);
    tg_bm[dst_tid] = _250;
    GroupMemoryBarrierWithGroupSync();
    s = 1u;
    src_tid = dst_tid ^ s;
    uint param_20 = dst_tid;
    uint param_21 = tg_bm[dst_tid];
    uint param_22 = tg_bm[src_tid];
    uint param_23 = 1431655765u;
    uint param_24 = s;
    uint _270 = shuffle_round(param_20, param_21, param_22, param_23, param_24);
    tg_bm[dst_tid] = _270;
    GroupMemoryBarrierWithGroupSync();
    if (dst_tid == 0u)
    {
        _62.Store(0, tg_bm[0]);
        _62.Store(4, tg_bm[1]);
        _62.Store(8, tg_bm[2]);
        _62.Store(12, tg_bm[3]);
        _62.Store(16, tg_bm[4]);
        _62.Store(20, tg_bm[5]);
        _62.Store(24, tg_bm[6]);
        _62.Store(28, tg_bm[7]);
        _62.Store(32, tg_bm[8]);
        _62.Store(36, tg_bm[9]);
        _62.Store(40, tg_bm[10]);
        _62.Store(44, tg_bm[11]);
        _62.Store(48, tg_bm[12]);
        _62.Store(52, tg_bm[13]);
        _62.Store(56, tg_bm[14]);
        _62.Store(60, tg_bm[15]);
        _62.Store(64, tg_bm[16]);
        _62.Store(68, tg_bm[17]);
        _62.Store(72, tg_bm[18]);
        _62.Store(76, tg_bm[19]);
        _62.Store(80, tg_bm[20]);
        _62.Store(84, tg_bm[21]);
        _62.Store(88, tg_bm[22]);
        _62.Store(92, tg_bm[23]);
        _62.Store(96, tg_bm[24]);
        _62.Store(100, tg_bm[25]);
        _62.Store(104, tg_bm[26]);
        _62.Store(108, tg_bm[27]);
        _62.Store(112, tg_bm[28]);
        _62.Store(116, tg_bm[29]);
        _62.Store(120, tg_bm[30]);
        _62.Store(124, tg_bm[31]);
    }
}

[numthreads(32, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
