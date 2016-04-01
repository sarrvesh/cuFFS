#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#ifdef __cplusplus
extern "C"
#endif

struct optionsList parseInput(char *parsetFileName);
void printOptions(struct optionsList inOptions);

#endif
