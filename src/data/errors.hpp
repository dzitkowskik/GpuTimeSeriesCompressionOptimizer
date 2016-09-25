//
//  errors.h
//  osx-system-data-logger
//
//  Created by Karol Dzitkowski on 03.12.2014.
//  Copyright (c) 2014 Karol Dzitkowski. All rights reserved.
//
#ifndef HELPER_ERRORS
#define HELPER_ERRORS

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifndef FILE_ERR
#define FILE_ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
perror(source),kill(0,SIGKILL),\
exit(EXIT_FAILURE))
#endif

#ifndef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(expression) \
({ \
long int _result; \
do _result = (long int) (expression); \
while (_result == -1L && errno == EINTR); \
_result; \
})
#endif

#ifndef CHECK_ERROR
#define CHECK_ERROR(expression) \
({ \
if(-1 == expression){ \
fprintf(stderr,"%s:%d\n",__FILE__,__LINE__); \
kill(0,SIGKILL); \
exit(EXIT_FAILURE);} \
})
#endif

#ifndef TEMP_FAILURE_RETRY_WHEN_NULL
#define TEMP_FAILURE_RETRY_WHEN_NULL(expression) \
({ \
void* _result; \
do _result = (void*) (expression); \
while (_result == NULL && errno == EINTR); \
_result; \
})
#endif

#ifndef CHECK_ERROR_WHEN_NULL
#define CHECK_ERROR_WHEN_NULL(expression) \
({ \
if(NULL == (void*)(expression) ){ \
fprintf(stderr,"%s:%d\n",__FILE__,__LINE__); \
kill(0,SIGKILL); \
exit(EXIT_FAILURE);} \
})
#endif

#endif // HELPER_ERRORS
