#!/bin/bash
# This script checks for the availability of AVX2 and SSE2 instructions

# Check for AVX2 support
if grep -q avx2 /proc/cpuinfo; then
  echo "-mavx2 -DUSE_AVX2"  # Output for Ninja to capture
  >&2 echo "Building for AVX2"  # Output for user, redirected to standard error
elif grep -q sse3 /proc/cpuinfo; then
  echo "msse3"
  >&2 echo "Building for SSE3"
elif grep -q sse2 /proc/cpuinfo; then
  echo "-msse2"  # Output for Ninja to capture
  >&2 echo "Building for SSE2"  # Output for user, redirected to standard error
else
  echo ""
  >&2 echo "Building without specific SIMD optimizations"
fi
