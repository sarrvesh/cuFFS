#ifndef SYNTHESIS_FILEACCESS_H
#define SYNTHESIS_FILEACCESS_H

/* Structure to store input options */
struct optionsList {
   char *msName;
};

struct optionsList parseInput(char *parsetFileName);

#endif
