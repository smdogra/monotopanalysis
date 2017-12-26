#include "../interface/PackingHelper.h"

#include <cmath>

panda::PackingHelper::PackingHelper()
{
  // ==== mantissatable ===
  // -- zero --
  mantissatable[0] = 0;
  // -- denorm --
  for (unsigned int i = 1; i <= 1023; ++i) {
    unsigned m = (i<<13);
    unsigned e = 0;
    while ((m & 0x00800000) == 0) { // While not normalized
      e -= 0x00800000; // Decrement exponent (1<<23)
      m <<= 1; // Shift mantissa
    }
    m &= ~0x00800000; // Clear leading 1 bit
    e += 0x38800000; // Adjust bias ((127-14)<<23)
    mantissatable[i] = m | e; 
  }
  // -- norm --
  for (unsigned int i = 1024; i <= 2047; ++i)
    mantissatable[i] = 0x38000000 + ((i - 1024) << 13);

  // ==== exponenttable ===
  exponenttable[0] = 0;
  for (unsigned int i = 1; i <= 30; ++i)
    exponenttable[i] = i << 23;
  exponenttable[31] = 0x47800000;
  exponenttable[32] = 0x80000000u;
  for (unsigned int i = 33; i <= 62; ++i)
    exponenttable[i] = 0x80000000u | ((i - 32) << 23);
  exponenttable[63] = 0xC7800000;
 
  // ==== offsettable ====
  for (unsigned int i = 0; i <= 63; ++i)
    offsettable[i] = ((i == 0 || i == 32) ? 0 : 1024);

  for (unsigned i = 0; i < 256; ++i) {
    int e = int(i) - 127;
    if (e < -24) { // Very small numbers map to zero
      basetable[i | 0x000] = 0x0000;
      basetable[i | 0x100] = 0x8000;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    }
    else if (e < -14) { // Small numbers map to denorms
      basetable[i | 0x000] = (0x0400 >> (-e - 14));
      basetable[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
      shifttable[i | 0x000] = -e - 1;
      shifttable[i | 0x100] = -e - 1;
    }
    else if (e <= 15) { // Normal numbers just lose precision
      basetable[i | 0x000] = ((e + 15) << 10);
      basetable[i | 0x100] = ((e + 15) << 10) | 0x8000;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    }
    else if (e < 128) { // Large numbers map to Infinity
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    }
    else { // Infinity and NaN's stay Infinity and NaN's
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    }
  }
}

UShort_t
panda::PackingHelper::packUnbound(Double_t x) const
{
  // Implementation copied from DataFormats/PatCandidates/interface/libminifloat.h
  // float32to16

  union {
    float flt;
    uint32_t i32;
  } conv;

  conv.flt = x;
  if (shifttable[(conv.i32>>23)&0x1ff] == 13) {
    uint16_t base2 = (conv.i32&0x007fffff)>>12;
    uint16_t base = base2 >> 1;
    if (((base2 & 1) != 0) && (base < 1023)) base++;
    return basetable[(conv.i32>>23)&0x1ff]+base; 
  }
  else
    return basetable[(conv.i32>>23)&0x1ff]+((conv.i32&0x007fffff)>>shifttable[(conv.i32>>23)&0x1ff]);
}

Double_t
panda::PackingHelper::unpackUnbound(UShort_t p) const
{
  union {
    float flt;
    uint32_t i32;
  } conv;

  conv.i32 = mantissatable[offsettable[p>>10]+(p&0x3ff)]+exponenttable[p>>10];
  return conv.flt;
}

Char_t
panda::PackingHelper::pack8LogBound(Double_t x, Double_t min, Double_t max, UChar_t baseminus1) const
{
  if (baseminus1 > 127)
    baseminus1 = 127;

  double l(std::log(std::abs(x)));
  Char_t r;
  if (l < min)
    r = 0;
  else if (l >= max)
    r = baseminus1;
  else
    r = std::round((l - min) / (max - min) * baseminus1);

  if (x < 0.) {
    if (r == 0)
      r = -1;
    else
      r *= -1;
  }
  return r;
}

Double_t
panda::PackingHelper::unpack8LogBound(Char_t i, Double_t min, Double_t max, UChar_t baseminus1) const
{
  if (baseminus1 > 127)
    baseminus1 = 127;

  double l;
  if (std::abs(i) == baseminus1)
    l = max;
  else
    l = min + std::abs(i) * (max - min) / baseminus1;

  double val(std::exp(l));
  if (i < 0)
    return -val;
  else
    return val;
}
