// Minimal definition of RCCL's getHostName(char*, int, char).
//
// In a Release librccl.so this symbol has hidden (internal) visibility, so a
// minimal MPI test binary that only links a subset of TUs against the prebuilt
// library gets an "undefined symbol: getHostName(char*, int, char)" at link
// time. This shim provides an ABI-compatible definition: copy the hostname and
// truncate it at the first occurrence of `delim` (matches upstream behaviour).
#include <unistd.h>
#include <cstring>

void getHostName(char* hostname, int maxlen, const char delim) {
  if (maxlen <= 0) return;
  if (gethostname(hostname, maxlen) != 0) {
    std::strncpy(hostname, "unknown", maxlen);
    hostname[maxlen - 1] = '\0';
    return;
  }
  hostname[maxlen - 1] = '\0';
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '\0') break;
    if (hostname[i] == delim) { hostname[i] = '\0'; break; }
  }
}
