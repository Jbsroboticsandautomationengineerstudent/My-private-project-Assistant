#import <Foundation/Foundation.h>

void HENewGame(BOOL playingWhite, int elo);
void HENoteOpponentMove(const char *uci);
void HEResetLines(void);
NSArray<NSString *> *HEConfigureCommands(int elo);
void HEParseInfo(const char *line);
NSString *HEChooseMove(const char *fen, int ply, int elo);
NSString *HEInferOpponentMove(const char *priorFen, const char *selectedMove,
                              const char *currentFen);
NSDictionary *HEDescribeElo(int elo);
