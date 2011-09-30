#ifndef LOG_H
#define LOG_H

#include <cstdlib>
#include <cstdio>

#define LOG_FATAL    (1)
#define LOG_ERR      (2)
#define LOG_WARN     (3)
#define LOG_INFO     (4)
#define LOG_DBG      (5)

const char *getLogString(int level) {
  switch (level) {
    case LOG_FATAL: return "FATAL";
    case LOG_ERR:   return "ERR  ";
    case LOG_WARN:  return "WARN ";
    case LOG_INFO:  return "INFO ";
    case LOG_DBG:   return "DEBUG";
    default:        return "-----";
  }
}

#define DEBUG_LEVEL LOG_WARN

#ifdef DEBUG_LEVEL
#define LOG(level, ...) do {                                                   \
  if (level <= DEBUG_LEVEL ) {                                                 \
    fprintf(stderr,"[%s] [%s:%d]: ", getLogString(level), __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
    fflush(stderr);                                                            \
  }                                                                            \
  if (level == LOG_FATAL) {                                                    \
    exit(1);                                                                   \
  }                                                                            \
} while (0)
#else
#define LOG(level, ...)  do { } while(0)
#endif

#endif
