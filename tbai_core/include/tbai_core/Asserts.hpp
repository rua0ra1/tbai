

#ifdef TBAI_DISABLE_ASSERTS
#define TBAI_ASSERT(condition)
#else

#define TBAI_ASSERT(condition)                                                                                    \
    do {                                                                                                          \
        if (!(condition)) {                                                                                       \
            std::cerr << "Assertion failed: " << #condition << " in file " << __FILE__ << " at line " << __LINE__ \
                      << std::endl;                                                                               \
            std::abort();                                                                                         \
        }                                                                                                         \
    } while (0)

#endif