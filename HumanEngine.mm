//
//  HumanEngine.mm
//  Chess Assistant Pro — human move simulation for ELO 600–1319
//
//  Every move returned by this module is a line that Stockfish itself searched
//  and scored (MultiPV). Nothing is invented. The module only decides *which*
//  of Stockfish's own moves to play, biased to look like a human of the target
//  rating: greedy, check-happy and less sensitive to deep tactics. The budget
//  and ELO labels are heuristics until measured against real games.
//
//  ELO 1320+ is untouched — fall through to UCI_LimitStrength / UCI_Elo.
//
//  ---------------------------------------------------------------------------
//  INTEGRATION (5 calls in the Stockfish 19 wrapper)
//  ---------------------------------------------------------------------------
//    1. game start / new game:
//           HENewGame(playingAsWhite, currentElo);
//
//    2. before each search, send these to the engine, then "go" is included:
//           for (NSString *c in HEConfigureCommands(currentElo)) sendUCI(c);
//           HEResetLines();
//
//    3. for every "info ..." line the engine prints:
//           HEParseInfo([line UTF8String]);
//
//    4. on "bestmove X":
//           NSString *m = HEChooseMove(fenCString, plyCount, currentElo);
//           forwardToEngineCallback(m ?: engineBestMove);
//    5. after inferring an opponent move from consecutive positions:
//           HENoteOpponentMove(opponentUCI);
//
//  plyCount = total half-moves played so far (0 at the start of the game).
//  fen      = position the engine just searched, standard FEN.
//
//  ---------------------------------------------------------------------------

#import <Foundation/Foundation.h>
#import "HumanEngine.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// ===========================================================================
// MARK: - Tunables
// ===========================================================================

#define HE_BAND_LO        600
#define HE_BAND_HI        1320      // exclusive
#define HE_MAX_PV         12
#define HE_MATE_CP        30000

// ===========================================================================
// MARK: - Band parameters (continuous across 600 → 1319)
// ===========================================================================

typedef struct {
    double t;               // 0.0 at 600, 1.0 at 1319
    int    skill;           // Stockfish Skill Level   0 → 3
    int    depth;           // search depth            5 → 10
    int    multipv;         // candidate lines         10 → 6
    double T;               // softmax temperature     145 → 45
    double seeThrough;      // tactical vision         0 → 1
    double blunderBudget;   // real blunders per game  4.5 → 1.0
    double mistakeBudget;   // mistakes per game       9 → 3
    int    convThresh;      // cp above which it always plays best  600 → 350
    double quietW;          // weight on quiet moves   0.50 → 0.95
    double greedBase, aggroBase, fearBase;
} HEParams;

static inline BOOL HEHumanBand(int elo) {
    return elo >= HE_BAND_LO && elo < HE_BAND_HI;
}

static HEParams HEParamsFor(int elo) {
    HEParams p;
    double t = (elo - (double)HE_BAND_LO) / 719.0;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    p.t = t;

    p.skill         = (int)lround(3.0 * t);
    p.depth         = (int)lround(5.0 + 5.0 * t);
    p.multipv       = (int)lround(10.0 - 4.0 * t);
    p.T             = 145.0 - 100.0 * t;
    p.seeThrough    = t;
    p.blunderBudget = 4.5 - 3.5 * t;
    p.mistakeBudget = 9.0 - 6.0 * t;
    p.convThresh    = (int)lround(600.0 - 250.0 * t);
    p.quietW        = 0.50 + 0.45 * t;
    p.greedBase     = 1.50 - 0.40 * t;
    p.aggroBase     = 1.45 - 0.35 * t;
    p.fearBase      = 0.40 + 0.40 * t;
    return p;
}

// ===========================================================================
// MARK: - RNG
// ===========================================================================

static inline double HEFrand(void) {
    return (double)arc4random_uniform(1000000u) / 1000000.0;
}

// ===========================================================================
// MARK: - Board (index 0 = a1 … 63 = h8)
// ===========================================================================

typedef struct {
    char sq[64];        // '.' = empty, uppercase = white, lowercase = black
    int  whiteToMove;
    int  epSquare;      // -1 if none
} HEBoard;

static inline int HEFile(int s) { return s & 7; }
static inline int HERank(int s) { return s >> 3; }
static inline int HEOnBoard(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
static inline int HESq(int f, int r) { return r * 8 + f; }
static inline int HEIsWhitePiece(char c) { return c != '.' && isupper((unsigned char)c); }

static int HEPieceValue(char p) {
    switch (tolower((unsigned char)p)) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        case 'k': return 10000;
        default:  return 0;
    }
}

static int HEParseSquare(const char *s) {
    if (!s || !s[0] || !s[1]) return -1;
    int f = s[0] - 'a';
    int r = s[1] - '1';
    if (!HEOnBoard(f, r)) return -1;
    return HESq(f, r);
}

// Parses the board, side to move and en-passant square. Castling rights and
// clocks are not needed by any heuristic here.
static BOOL HEParseFEN(const char *fen, HEBoard *out) {
    if (!fen || !out) return NO;
    memset(out->sq, '.', sizeof(out->sq));
    out->whiteToMove = 1;
    out->epSquare    = -1;

    int rank = 7, file = 0;
    const char *p = fen;

    for (; *p && *p != ' '; p++) {
        if (*p == '/') {
            if (file != 8 || rank == 0) return NO;
            rank--; file = 0;
            continue;
        }
        if (*p >= '1' && *p <= '8') {
            file += *p - '0';
            if (file > 8) return NO;
            continue;
        }
        if (file > 7 || !strchr("PNBRQKpnbrqk", *p)) return NO;
        out->sq[HESq(file, rank)] = *p;
        file++;
    }
    if (rank != 0 || file != 8) return NO;

    while (*p == ' ') p++;
    if (*p != 'w' && *p != 'b') return NO;
    out->whiteToMove = (*p == 'w');
    p++;

    while (*p == ' ') p++;              // skip side-to-move field remainder
    while (*p && *p != ' ') p++;        // castling rights
    while (*p == ' ') p++;
    if (*p && *p != '-') out->epSquare = HEParseSquare(p);

    return YES;
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------

static const int HEKnightD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
static const int HEKingD[8][2]   = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};
static const int HEBishopD[4][2] = {{1,1},{1,-1},{-1,-1},{-1,1}};
static const int HERookD[4][2]   = {{0,1},{1,0},{0,-1},{-1,0}};

static int HEAttackedBy(const HEBoard *b, int sq, int byWhite) {
    int f = HEFile(sq), r = HERank(sq);

    // pawns
    if (byWhite) {
        if (HEOnBoard(f - 1, r - 1) && b->sq[HESq(f - 1, r - 1)] == 'P') return 1;
        if (HEOnBoard(f + 1, r - 1) && b->sq[HESq(f + 1, r - 1)] == 'P') return 1;
    } else {
        if (HEOnBoard(f - 1, r + 1) && b->sq[HESq(f - 1, r + 1)] == 'p') return 1;
        if (HEOnBoard(f + 1, r + 1) && b->sq[HESq(f + 1, r + 1)] == 'p') return 1;
    }

    // knights
    for (int i = 0; i < 8; i++) {
        int nf = f + HEKnightD[i][0], nr = r + HEKnightD[i][1];
        if (!HEOnBoard(nf, nr)) continue;
        char c = b->sq[HESq(nf, nr)];
        if (c == (byWhite ? 'N' : 'n')) return 1;
    }

    // king
    for (int i = 0; i < 8; i++) {
        int nf = f + HEKingD[i][0], nr = r + HEKingD[i][1];
        if (!HEOnBoard(nf, nr)) continue;
        char c = b->sq[HESq(nf, nr)];
        if (c == (byWhite ? 'K' : 'k')) return 1;
    }

    // bishops / queens
    for (int i = 0; i < 4; i++) {
        int nf = f, nr = r;
        for (;;) {
            nf += HEBishopD[i][0]; nr += HEBishopD[i][1];
            if (!HEOnBoard(nf, nr)) break;
            char c = b->sq[HESq(nf, nr)];
            if (c == '.') continue;
            if (byWhite ? (c == 'B' || c == 'Q') : (c == 'b' || c == 'q')) return 1;
            break;
        }
    }

    // rooks / queens
    for (int i = 0; i < 4; i++) {
        int nf = f, nr = r;
        for (;;) {
            nf += HERookD[i][0]; nr += HERookD[i][1];
            if (!HEOnBoard(nf, nr)) break;
            char c = b->sq[HESq(nf, nr)];
            if (c == '.') continue;
            if (byWhite ? (c == 'R' || c == 'Q') : (c == 'r' || c == 'q')) return 1;
            break;
        }
    }

    return 0;
}

// Cheapest enemy piece attacking sq (HE_MATE_CP if none).
static int HECheapestAttacker(const HEBoard *b, int sq, int byWhite) {
    int best = HE_MATE_CP;
    int f = HEFile(sq), r = HERank(sq);

    if (byWhite) {
        if (HEOnBoard(f - 1, r - 1) && b->sq[HESq(f - 1, r - 1)] == 'P') best = 100;
        if (HEOnBoard(f + 1, r - 1) && b->sq[HESq(f + 1, r - 1)] == 'P') best = 100;
    } else {
        if (HEOnBoard(f - 1, r + 1) && b->sq[HESq(f - 1, r + 1)] == 'p') best = 100;
        if (HEOnBoard(f + 1, r + 1) && b->sq[HESq(f + 1, r + 1)] == 'p') best = 100;
    }
    if (best == 100) return 100;

    for (int i = 0; i < 8; i++) {
        int nf = f + HEKnightD[i][0], nr = r + HEKnightD[i][1];
        if (!HEOnBoard(nf, nr)) continue;
        if (b->sq[HESq(nf, nr)] == (byWhite ? 'N' : 'n')) { if (320 < best) best = 320; }
    }

    for (int i = 0; i < 4; i++) {
        int nf = f, nr = r;
        for (;;) {
            nf += HEBishopD[i][0]; nr += HEBishopD[i][1];
            if (!HEOnBoard(nf, nr)) break;
            char c = b->sq[HESq(nf, nr)];
            if (c == '.') continue;
            if (byWhite ? (c == 'B') : (c == 'b')) { if (330 < best) best = 330; }
            else if (byWhite ? (c == 'Q') : (c == 'q')) { if (900 < best) best = 900; }
            break;
        }
    }

    for (int i = 0; i < 4; i++) {
        int nf = f, nr = r;
        for (;;) {
            nf += HERookD[i][0]; nr += HERookD[i][1];
            if (!HEOnBoard(nf, nr)) break;
            char c = b->sq[HESq(nf, nr)];
            if (c == '.') continue;
            if (byWhite ? (c == 'R') : (c == 'r')) { if (500 < best) best = 500; }
            else if (byWhite ? (c == 'Q') : (c == 'q')) { if (900 < best) best = 900; }
            break;
        }
    }

    for (int i = 0; i < 8; i++) {
        int nf = f + HEKingD[i][0], nr = r + HEKingD[i][1];
        if (!HEOnBoard(nf, nr)) continue;
        if (b->sq[HESq(nf, nr)] == (byWhite ? 'K' : 'k')) { if (10000 < best) best = 10000; }
    }

    return best;
}

static int HEFindKing(const HEBoard *b, int white) {
    char k = white ? 'K' : 'k';
    for (int i = 0; i < 64; i++) if (b->sq[i] == k) return i;
    return -1;
}

// ---------------------------------------------------------------------------
// Move application (enough fidelity for the heuristics)
// ---------------------------------------------------------------------------

typedef struct { int from, to; char promo; } HEMove;

static HEMove HEParseMove(const char *uci) {
    HEMove m = { -1, -1, 0 };
    if (!uci || strlen(uci) < 4) return m;
    m.from  = HEParseSquare(uci);
    m.to    = HEParseSquare(uci + 2);
    m.promo = (strlen(uci) >= 5) ? (char)tolower((unsigned char)uci[4]) : 0;
    return m;
}

static HEBoard HEApply(const HEBoard *in, HEMove m) {
    HEBoard b = *in;
    if (m.from < 0 || m.to < 0) return b;

    char piece = b.sq[m.from];
    char lower = (char)tolower((unsigned char)piece);
    int  white = HEIsWhitePiece(piece);

    // en passant capture
    if (lower == 'p' && m.to == in->epSquare && b.sq[m.to] == '.') {
        int capSq = white ? m.to - 8 : m.to + 8;
        if (capSq >= 0 && capSq < 64) b.sq[capSq] = '.';
    }

    // castling: move the rook too
    if (lower == 'k' && abs(HEFile(m.to) - HEFile(m.from)) == 2) {
        int r = HERank(m.from);
        if (HEFile(m.to) == 6) {                       // king side
            b.sq[HESq(5, r)] = b.sq[HESq(7, r)];
            b.sq[HESq(7, r)] = '.';
        } else if (HEFile(m.to) == 2) {                // queen side
            b.sq[HESq(3, r)] = b.sq[HESq(0, r)];
            b.sq[HESq(0, r)] = '.';
        }
    }

    b.sq[m.to]   = m.promo ? (white ? (char)toupper((unsigned char)m.promo) : m.promo) : piece;
    b.sq[m.from] = '.';

    // new en-passant square
    b.epSquare = -1;
    if (lower == 'p' && abs(HERank(m.to) - HERank(m.from)) == 2)
        b.epSquare = white ? m.from + 8 : m.from - 8;

    b.whiteToMove = !in->whiteToMove;
    return b;
}

// Infer the opponent's actual move when the next search follows a move we
// selected. Comparing resulting boards avoids guessing from square changes in
// castling, en passant and promotion positions.
NSString *HEInferOpponentMove(const char *priorFen, const char *selectedMove,
                              const char *currentFen) {
    HEBoard prior, current;
    if (!HEParseFEN(priorFen, &prior) || !HEParseFEN(currentFen, &current)) return nil;
    HEMove ours = HEParseMove(selectedMove);
    if (ours.from < 0 || ours.to < 0 || prior.sq[ours.from] == '.' ||
        HEIsWhitePiece(prior.sq[ours.from]) != prior.whiteToMove ||
        current.whiteToMove != prior.whiteToMove) return nil;
    HEBoard afterOurs = HEApply(&prior, ours);
    NSString *match = nil;
    for (int from = 0; from < 64; ++from) {
        char piece = afterOurs.sq[from];
        if (piece == '.' || HEIsWhitePiece(piece) != afterOurs.whiteToMove) continue;
        for (int to = 0; to < 64; ++to) {
            if (from == to) continue;
            BOOL promotion = tolower((unsigned char)piece) == 'p' &&
                HERank(to) == (afterOurs.whiteToMove ? 7 : 0);
            const char *promotions = promotion ? "qrbn" : "-";
            for (const char *promo = promotions; *promo; ++promo) {
                HEMove candidate = {from, to, (char)(promotion ? *promo : 0)};
                HEBoard result = HEApply(&afterOurs, candidate);
                if (result.whiteToMove != current.whiteToMove ||
                    memcmp(result.sq, current.sq, sizeof(result.sq)) != 0) continue;
                NSString *uci = [NSString stringWithFormat:@"%c%c%c%c%@",
                    'a' + HEFile(from), '1' + HERank(from),
                    'a' + HEFile(to), '1' + HERank(to),
                    promotion ? [NSString stringWithFormat:@"%c", *promo] : @""];
                if (match) return nil; // Ambiguous: never record a guessed move.
                match = uci;
            }
        }
    }
    return match;
}

// ===========================================================================
// MARK: - Move features
// ===========================================================================

typedef struct {
    int isCapture, isRecapture, capturesHigher, givesCheck, attacksQueen;
    int isBackward, isQuiet, isSamePieceAgain, isPawnPush, hangsPiece;
    int isDeveloping, isCastle, isEarlyQueenSortie, isRookPawn, isPromotion;
    int visibleMistake;
} HEFeatures;

static int HELastOwnTo   = -1;   // destination of our previous move
static int HEOppLastTo   = -1;   // destination of the opponent's last move

static HEFeatures HEExtract(const HEBoard *b, HEMove m) {
    HEFeatures f;
    memset(&f, 0, sizeof(f));
    if (m.from < 0 || m.to < 0) return f;

    char piece = b->sq[m.from];
    char lower = (char)tolower((unsigned char)piece);
    int  white = HEIsWhitePiece(piece);
    char target = b->sq[m.to];

    f.isCapture      = (target != '.') ||
                       (lower == 'p' && m.to == b->epSquare);
    f.isRecapture    = f.isCapture && (m.to == HEOppLastTo);
    f.capturesHigher = f.isCapture && target != '.' &&
                       HEPieceValue(target) > HEPieceValue(piece);
    f.isPromotion    = (m.promo != 0);
    f.isPawnPush     = (lower == 'p') && !f.isCapture;
    f.isRookPawn     = (lower == 'p') && (HEFile(m.from) == 0 || HEFile(m.from) == 7);
    f.isCastle       = (lower == 'k') && abs(HEFile(m.to) - HEFile(m.from)) == 2;
    f.isSamePieceAgain = (m.from == HELastOwnTo);

    int dr = HERank(m.to) - HERank(m.from);
    f.isBackward = white ? (dr < 0) : (dr > 0);

    int homeRank = white ? 0 : 7;
    f.isDeveloping = (lower == 'n' || lower == 'b') && (HERank(m.from) == homeRank);
    f.isEarlyQueenSortie = (lower == 'q') && (HERank(m.from) == homeRank);

    HEBoard after = HEApply(b, m);

    int oppKing = HEFindKing(&after, !white);
    f.givesCheck = (oppKing >= 0) && HEAttackedBy(&after, oppKing, white);

    char oq = white ? 'q' : 'Q';
    for (int i = 0; i < 64; i++) {
        if (after.sq[i] != oq) continue;
        if (HEAttackedBy(&after, i, white)) f.attacksQueen = 1;
        break;
    }

    f.isQuiet = !f.isCapture && !f.givesCheck && !f.isPromotion && !f.attacksQueen;

    // SEE-lite on the destination square
    int movedValue  = HEPieceValue(after.sq[m.to]);
    int attackedBy  = HEAttackedBy(&after, m.to, !white);
    int defended    = HEAttackedBy(&after, m.to, white);
    int cheapest    = attackedBy ? HECheapestAttacker(&after, m.to, !white) : HE_MATE_CP;
    int captured    = (target != '.') ? HEPieceValue(target) : 0;

    if (attackedBy && (cheapest < movedValue || !defended)) {
        int net = captured - movedValue;
        if (net < -80) f.hangsPiece = 1;
    }

    // A mistake is "visible" to a weak player when the refutation is one move
    // deep: the piece just moved can be taken for profit, or a piece was left
    // hanging that is now attacked by something cheaper.
    f.visibleMistake = f.hangsPiece;
    if (!f.visibleMistake) {
        char own = white ? 'A' : 'a';
        (void)own;
        for (int i = 0; i < 64; i++) {
            char c = after.sq[i];
            if (c == '.' || HEIsWhitePiece(c) != white) continue;
            if (tolower((unsigned char)c) == 'p' || tolower((unsigned char)c) == 'k') continue;
            if (i == m.to) continue;
            if (!HEAttackedBy(&after, i, !white)) continue;
            int ca = HECheapestAttacker(&after, i, !white);
            if (ca < HEPieceValue(c) && !HEAttackedBy(&after, i, white)) {
                f.visibleMistake = 1;
                break;
            }
        }
    }

    return f;
}

// ---------------------------------------------------------------------------
// Position descriptors
// ---------------------------------------------------------------------------

typedef enum { HEOpening = 0, HEMiddlegame = 1, HEEndgame = 2 } HEPhase;

static int HEPieceCount(const HEBoard *b) {
    int n = 0;
    for (int i = 0; i < 64; i++) if (b->sq[i] != '.') n++;
    return n;
}

static int HENonPawnMaterial(const HEBoard *b) {
    int v = 0;
    for (int i = 0; i < 64; i++) {
        char c = b->sq[i];
        if (c == '.') continue;
        char l = (char)tolower((unsigned char)c);
        if (l == 'p' || l == 'k') continue;
        v += HEPieceValue(c);
    }
    return v;
}

static HEPhase HEPhaseOf(const HEBoard *b, int ply) {
    int npm = HENonPawnMaterial(b);
    if (npm <= 2200) return HEEndgame;
    if (ply < 16 && npm >= 5600) return HEOpening;
    return HEMiddlegame;
}

// How tangled the position is — drives when blunders happen.
static double HEComplexity(const HEBoard *b, int white) {
    int hits = 0, nearKing = 0;
    for (int i = 0; i < 64; i++) {
        char c = b->sq[i];
        if (c == '.' || HEIsWhitePiece(c) == white) continue;
        if (HEAttackedBy(b, i, white)) hits++;
    }
    int ok = HEFindKing(b, !white);
    if (ok >= 0) {
        int f = HEFile(ok), r = HERank(ok);
        for (int i = 0; i < 8; i++) {
            int nf = f + HEKingD[i][0], nr = r + HEKingD[i][1];
            if (!HEOnBoard(nf, nr)) continue;
            if (HEAttackedBy(b, HESq(nf, nr), white)) nearKing++;
        }
    }
    return hits / 6.0 + nearKing / 4.0;
}

// ===========================================================================
// MARK: - Per-game persona
// ===========================================================================

typedef struct {
    double greed, aggro, fear;
    double blunderBudget, mistakeBudget;
    int    bookChoice;
    int    isWhite;
    int    init;
} HEPersona;

static HEPersona HEP;
static HEParams  HEHP;

void HENewGame(BOOL playingWhite, int elo) {
    HEHP = HEParamsFor(elo);

    HEP.isWhite = playingWhite ? 1 : 0;
    HEP.greed   = HEHP.greedBase + HEFrand() * 0.50;
    HEP.aggro   = HEHP.aggroBase + HEFrand() * 0.45;
    HEP.fear    = HEHP.fearBase  + HEFrand() * 0.30;

    HEP.blunderBudget = HEHP.blunderBudget + (HEFrand() - 0.5);
    HEP.mistakeBudget = HEHP.mistakeBudget + (HEFrand() * 2.0 - 1.0);

    HEP.bookChoice = (int)arc4random_uniform(1000);

    // white plays for the initiative, black reacts
    if (playingWhite) HEP.aggro *= 1.25; else HEP.fear *= 1.18;

    HELastOwnTo = -1;
    HEOppLastTo = -1;
    HEP.init    = 1;
}

// Call when the opponent moves, so recapture / same-piece logic works.
void HENoteOpponentMove(const char *uci) {
    HEMove m = HEParseMove(uci);
    HEOppLastTo = m.to;
}

// ===========================================================================
// MARK: - UCI plumbing
// ===========================================================================

typedef struct {
    int  cp;        // score from the engine's point of view, centipawns
    int  mateIn;    // 0 = none, >0 = we mate in N, <0 = we get mated in N
    int  valid;
    char move[8];
} HELine;

static HELine HELines[HE_MAX_PV];
static int    HELineCount = 0;
static int    HELineDepth = 0;
static HELine HEPreviousLines[HE_MAX_PV];
static int    HEPreviousLineCount = 0;

void HEResetLines(void) {
    memset(HELines, 0, sizeof(HELines));
    HELineCount = 0;
    HELineDepth = 0;
    memset(HEPreviousLines, 0, sizeof(HEPreviousLines));
    HEPreviousLineCount = 0;
}

NSArray<NSString *> *HEConfigureCommands(int elo) {
    NSMutableArray<NSString *> *out = [NSMutableArray array];

    if (!HEHumanBand(elo)) {
        [out addObject:@"setoption name MultiPV value 1"];
        [out addObject:@"setoption name Skill Level value 20"];
        if (elo < 3190) {
            [out addObject:@"setoption name UCI_LimitStrength value true"];
            [out addObject:[NSString stringWithFormat:@"setoption name UCI_Elo value %d", elo]];
        } else {
            [out addObject:@"setoption name UCI_LimitStrength value false"];
        }
        [out addObject:@"go movetime 200"];
        return out;
    }

    if (!HEP.init) HENewGame(YES, elo);          // failsafe
    HEHP = HEParamsFor(elo);

    [out addObject:@"setoption name UCI_LimitStrength value false"];
    [out addObject:@"setoption name Threads value 1"];
    [out addObject:[NSString stringWithFormat:@"setoption name Skill Level value %d", HEHP.skill]];
    [out addObject:[NSString stringWithFormat:@"setoption name MultiPV value %d", HEHP.multipv]];

    int depth = HEHP.depth + (int)arc4random_uniform(3);
    int ms    = 110 + (int)arc4random_uniform(140);
    [out addObject:[NSString stringWithFormat:@"go depth %d movetime %d", depth, ms]];

    return out;
}

// Parses one "info ... multipv K ... score cp X ... pv MOVE ..." line.
void HEParseInfo(const char *line) {
    if (!line) return;
    if (!strstr(line, " multipv ") || !strstr(line, " pv ")) return;

    const char *depthField = strstr(line, " depth ");
    if (!depthField) return;
    int depth = atoi(depthField + 7);
    if (depth <= 0 || depth < HELineDepth) return;
    if (depth > HELineDepth) {
        if (HELineCount >= 2 && HELines[0].valid) {
            memcpy(HEPreviousLines, HELines, sizeof(HELines));
            HEPreviousLineCount = HELineCount;
        }
        memset(HELines, 0, sizeof(HELines));
        HELineCount = 0;
        HELineDepth = depth;
    }

    int idx = 0, cp = 0, mate = 0, haveScore = 0;
    char mv[8] = {0};

    const char *p = strstr(line, " multipv ");
    if (p) idx = atoi(p + 9);
    if (idx < 1 || idx > HE_MAX_PV) return;

    p = strstr(line, " score cp ");
    if (p) { cp = atoi(p + 10); haveScore = 1; }

    p = strstr(line, " score mate ");
    if (p) {
        mate = atoi(p + 12);
        cp = (mate > 0) ? (HE_MATE_CP - mate) : (-HE_MATE_CP - mate);
        haveScore = 1;
    }
    if (!haveScore) return;

    p = strstr(line, " pv ");
    if (!p) return;
    p += 4;
    int n = 0;
    while (*p && *p != ' ' && n < 7) mv[n++] = *p++;
    mv[n] = 0;
    if ((n != 4 && n != 5) || HEParseSquare(mv) < 0 ||
        HEParseSquare(mv + 2) < 0 ||
        (n == 5 && !strchr("qrbn", mv[4]))) return;

    HELine *l = &HELines[idx - 1];
    l->cp     = cp;
    l->mateIn = mate;
    l->valid  = 1;
    strncpy(l->move, mv, sizeof(l->move) - 1);
    l->move[sizeof(l->move) - 1] = 0;

    if (idx > HELineCount) HELineCount = idx;
}

// ===========================================================================
// MARK: - Human bias
// ===========================================================================

static double HEHumanBias(const HEFeatures *f, int ply) {
    double t = HEHP.t, inv = 1.0 - t, w = 1.0;

    if (f->isCapture)        w *= HEP.greed;
    if (f->isRecapture)      w *= 1.0 + 0.90 * inv;
    if (f->capturesHigher)   w *= 1.0 + 0.60 * inv;
    if (f->givesCheck)       w *= HEP.aggro;
    if (f->attacksQueen)     w *= 1.0 + 0.30 * inv;
    if (f->isPromotion)      w *= 1.6;

    if (f->isBackward)       w *= HEP.fear;
    if (f->isQuiet)          w *= HEHP.quietW;
    if (f->isSamePieceAgain) w *= 1.0 + 0.35 * inv;
    if (f->isPawnPush)       w *= 1.0 + 0.20 * inv;
    if (f->hangsPiece)       w *= 0.25 + 0.60 * t;

    if (ply < 16) {
        if (f->isDeveloping)       w *= 1.15 + 0.35 * t;
        if (f->isCastle)           w *= 0.65 + 0.45 * t;
        if (f->isEarlyQueenSortie) w *= 1.0 + 0.55 * inv;
        if (f->isRookPawn)         w *= 0.45 + 0.45 * t;
    }

    if (w < 0.01) w = 0.01;
    return w;
}

// ===========================================================================
// MARK: - Move choice
// ===========================================================================

// Returns the move to play, or nil when the ELO is outside the human band
// (caller should then use the engine's own bestmove).
static NSString *HECommitMove(const HELine *lines, int index) {
    HEMove m = HEParseMove(lines[index].move);
    HELastOwnTo = m.to;
    return [NSString stringWithUTF8String:lines[index].move];
}

NSString *HEChooseMove(const char *fen, int ply, int elo) {
    if (!HEHumanBand(elo)) return nil;
    const HELine *source = HELines;
    int sourceCount = HELineCount;
    if (sourceCount < 2 && HEPreviousLineCount >= 2) {
        source = HEPreviousLines;
        sourceCount = HEPreviousLineCount;
    }
    if (sourceCount < 1 || !source[0].valid) return nil;

    HEHP = HEParamsFor(elo);
    if (!HEP.init) HENewGame(ply % 2 == 0, elo);

    // compact the valid lines
    HELine L[HE_MAX_PV];
    int n = 0;
    for (int i = 0; i < sourceCount && i < HE_MAX_PV; i++)
        if (source[i].valid) L[n++] = source[i];
    if (n == 0) return nil;

    HEBoard b;
    int haveBoard = HEParseFEN(fen, &b) ? 1 : 0;
    int white     = haveBoard ? b.whiteToMove : HEP.isWhite;
    int best      = L[0].cp;

    // -------------------------------------------------------------------
    // 1. Play to win — never err where the game is decided
    // -------------------------------------------------------------------
    if (best > HEHP.convThresh)                           return HECommitMove(L, 0);
    if (L[0].mateIn > 0 && L[0].mateIn <= 3 + (int)(3 * HEHP.t)) return HECommitMove(L, 0);
    if (best < -500)                                      return HECommitMove(L, 0);

    HEPhase ph = haveBoard ? HEPhaseOf(&b, ply) : HEMiddlegame;
    if (haveBoard && ph == HEEndgame &&
        HEPieceCount(&b) <= 7 + (int)(3 * HEHP.t))        return HECommitMove(L, 0);

    // -------------------------------------------------------------------
    // 2. Opening variety — a different game every time, still Stockfish moves
    // -------------------------------------------------------------------
    if (ply <= 10 && n >= 3) {
        int pool = (ply < 4) ? 6 - (int)(2 * HEHP.t) : 4 - (int)HEHP.t;
        if (pool > n) pool = n;
        if (pool < 2) pool = 2;
        int pick = (HEP.bookChoice + ply * 7 + (int)arc4random_uniform((uint32_t)pool)) % pool;
        if (best - L[pick].cp <= 120) return HECommitMove(L, pick);
    }

    // -------------------------------------------------------------------
    // 3. Spend the blunder budget, concentrated in complex middlegames
    // -------------------------------------------------------------------
    double complexity = haveBoard ? HEComplexity(&b, white) : 0.6;
    double rate = 0.0;
    if (HEP.blunderBudget > 0.0 && ph == HEMiddlegame) {
        rate = (0.045 + 0.025 * (1.0 - HEHP.t)) * fmin(2.0, 0.6 + complexity);
        if (ply < 12) rate *= 0.3;
    }

    if (n >= 4 && HEFrand() < rate) {
        int k = n - 1 - (int)arc4random_uniform(2);
        if (k < 1) k = n - 1;
        HEFeatures bf = {};
        if (haveBoard) bf = HEExtract(&b, HEParseMove(L[k].move));
        // a human blunder is one they cannot see; obvious ones are rarer
        if (!bf.visibleMistake || HEFrand() < 0.40 - 0.25 * HEHP.t) {
            HEP.blunderBudget -= 1.0;
            return HECommitMove(L, k);
        }
    }

    // -------------------------------------------------------------------
    // 4. Normal human sampling over Stockfish's own lines
    // -------------------------------------------------------------------
    double w[HE_MAX_PV], sum = 0.0;

    for (int i = 0; i < n; i++) {
        int loss = best - L[i].cp;
        if (loss < 0) loss = 0;

        HEFeatures f;
        if (haveBoard) f = HEExtract(&b, HEParseMove(L[i].move));
        else           memset(&f, 0, sizeof(f));

        double perceived = f.visibleMistake ? (double)loss : loss * 0.30;
        perceived = loss * HEHP.seeThrough + perceived * (1.0 - HEHP.seeThrough);

        if (loss > 200) {
            if (HEP.mistakeBudget <= 0.0) { w[i] = 0.0; continue; }
            perceived *= 0.75;
        }

        w[i] = exp(-perceived / HEHP.T) * HEHumanBias(&f, ply);

        if (loss > 700 && f.visibleMistake) w[i] *= 0.10;
        if (L[i].mateIn < 0)                w[i] *= 0.05;   // don't walk into mate

        sum += w[i];
    }

    if (sum <= 0.0) return HECommitMove(L, 0);

    double r = HEFrand() * sum, acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += w[i];
        if (r <= acc) {
            if (best - L[i].cp > 200) HEP.mistakeBudget -= 1.0;
            return HECommitMove(L, i);
        }
    }

    return HECommitMove(L, 0);
}

// ===========================================================================
// MARK: - Introspection (for your canvas / debug overlay)
// ===========================================================================

NSDictionary *HEDescribeElo(int elo) {
    if (!HEHumanBand(elo)) {
        return @{ @"mode"  : (elo < 3190 ? @"Stockfish UCI ELO limit"
                                         : @"Full strength · no fixed ELO"),
                  @"skill" : @20 };
    }
    HEParams p = HEParamsFor(elo);
    return @{
        @"mode"     : [NSString stringWithFormat:
                        @"Human sim · skill %d · depth %d · MultiPV %d · ~%.1f blunders",
                        p.skill, p.depth, p.multipv, p.blunderBudget],
        @"skill"    : @(p.skill),
        @"depth"    : @(p.depth),
        @"multipv"  : @(p.multipv),
        @"temp"     : @(p.T),
        @"blunders" : @(p.blunderBudget),
        @"mistakes" : @(p.mistakeBudget),
        @"convCp"   : @(p.convThresh)
    };
}
