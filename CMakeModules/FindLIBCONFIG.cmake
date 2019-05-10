if(NOT LIBCONFIG_FOUND)
    find_path(LIBCONFIG_INCLUDES
        NAMES libconfig.h
        HINTS ${LIBCONFIG_ROOT_DIR} $ENV{LIBCONFIG}
        PATH_SUFFIXES include)
    find_library(LIBCONFIG_LIBRARIES
        config
        HINTS ${LIBCONFIG_ROOT_DIR} $ENV{LIBCONFIG}
        PATH_SUFFIXES lib)
    
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(LIBCONFIG DEFAULT_MSG 
        LIBCONFIG_INCLUDES LIBCONFIG_LIBRARIES)
    mark_as_advanced(LIBCONFIG_INCLUDES LIBCONFIG_LIBRARIES)
endif(NOT LIBCONFIG_FOUND)
