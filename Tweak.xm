#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>
#include <onnxruntime_c_api.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <substrate.h>
#import <dlfcn.h>
#import <dispatch/dispatch.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

static NSString *const kMaiaKey = @"isMaiaEnabled";
static NSString *const kStockfishKey = @"isStockfishEnabled";
static NSString *const kOriginalMaiaKey = @"ChessAssist_UseMaia";
static NSString *const kMaiaEloKey = @"ChessAssistantPro_MaiaELO";
static NSString *const kStockfishStrengthKey = @"ChessAssistantPro_StockfishStrength";
static NSString *const kMaiaAutoKey = @"ChessAssistantPro_MaiaAuto";
static NSString *const kStockfishAutoKey = @"ChessAssistantPro_StockfishAuto";
static NSString *const kModelPath = @"/var/jb/Library/Application Support/ChessAssistantPro/maia3_79m_int32.onnx";
static char kPanelKey;
static char kArrowLayerKey;
static char kScreenKey;
static NSString *currentFEN = nil;
static NSString *openingName = nil;
static NSString *maiaMove = nil;
static NSString *stockfishMove = nil;
static NSString *lastAutoPlayedFEN = nil;
static NSString *predictionFEN = nil;
static double maiaWDL[3] = {0, 0, 0};
static double stockfishWDL[3] = {0, 0, 0};
static BOOL hasMaiaWDL = NO;
static BOOL hasStockfishWDL = NO;
static NSHashTable<UIView *> *visiblePanels = nil;
static NSLock *CAPFenLock(void) {
    static NSLock *lock;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ lock = [NSLock new]; });
    return lock;
}

static void CAPRefreshAll(void);
static void CAPRememberFEN(NSString *fen);
static void CAPRememberMove(BOOL stockfish, NSString *move, NSString *fen);

// Recovered from the binary's callback encodings. The fields after move are
// intentionally opaque; the zero result means "no move" to the original UI.
struct EngineLine { char move[8]; bool flag1; bool flag2; int value; };
struct MaiaResult { char move[8]; double value1; double value2; bool flag; int value3; char moves[4][8]; double values[4]; };
static_assert(sizeof(EngineLine) == 16, "EngineLine ABI changed");
static_assert(sizeof(MaiaResult) == 96, "MaiaResult ABI changed");
typedef void (^EngineCallback)(const EngineLine *, int);
typedef void (^MaiaCallback)(MaiaResult);
static void CAPStartStockfish(const char *fen, uintptr_t arg1, uintptr_t arg2,
                              uintptr_t arg3, EngineCallback callback);
typedef void (*EngineGoFn)(const char *, uintptr_t, uintptr_t, uintptr_t, EngineCallback);
typedef void (*MaiaGoFn)(const char *, uintptr_t, uintptr_t, MaiaCallback);
static EngineGoFn originalEngineGo = nullptr;
static MaiaGoFn originalMaiaGo = nullptr;

static BOOL CAPMaiaEnabled(void) { return [[NSUserDefaults standardUserDefaults] boolForKey:kMaiaKey]; }
static BOOL CAPStockfishEnabled(void) { return [[NSUserDefaults standardUserDefaults] boolForKey:kStockfishKey]; }
static BOOL CAPMaiaAuto(void) { return [[NSUserDefaults standardUserDefaults] boolForKey:kMaiaAutoKey]; }
static BOOL CAPStockfishAuto(void) { return [[NSUserDefaults standardUserDefaults] boolForKey:kStockfishAutoKey]; }
static NSInteger CAPMaiaElo(void) {
    NSInteger value = [[NSUserDefaults standardUserDefaults] integerForKey:kMaiaEloKey];
    return value ? MAX(1100, MIN(2700, value)) : 1500;
}
static NSInteger CAPStockfishStrength(void) {
    NSInteger value = [[NSUserDefaults standardUserDefaults] integerForKey:kStockfishStrengthKey];
    return value ? MAX(1320, MIN(3200, value)) : 2300;
}
static int CAPStockfishDepth(void) {
    NSInteger strength = CAPStockfishStrength();
    return strength >= 3200 ? 30 : 8 + (int)((strength - 1320) * 22 / 1880);
}

static void CAPRemoveMarkedLayers(CALayer *parent) {
    for (CALayer *child in [parent.sublayers copy]) {
        if ([objc_getAssociatedObject(child, &kArrowLayerKey) boolValue]) [child removeFromSuperlayer];
        else CAPRemoveMarkedLayers(child);
    }
}

static void CAPSetMode(BOOL maia, BOOL stockfish) {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setBool:maia forKey:kMaiaKey];
    [defaults setBool:stockfish forKey:kStockfishKey];
    [defaults setBool:maia forKey:kOriginalMaiaKey];
    if (!maia) { maiaMove = nil; hasMaiaWDL = NO; }
    if (!stockfish) { stockfishMove = nil; hasStockfishWDL = NO; }
    if (!maia && !stockfish) lastAutoPlayedFEN = nil;
    CAPRefreshAll();
}

static NSString *CAPOpeningForFEN(NSString *fen) {
    @try {
        Class cls = NSClassFromString(@"CHOpeningBookManager");
        if (!cls || ![cls respondsToSelector:@selector(sharedOpeningBookManager)]) return @"Unknown";
        id manager = ((id (*)(id, SEL))objc_msgSend)(cls, @selector(sharedOpeningBookManager));
        if (![manager respondsToSelector:@selector(openingNameByFEN:)]) return @"Unknown";
        id result = ((id (*)(id, SEL, id))objc_msgSend)(manager, @selector(openingNameByFEN:), fen);
        return [result isKindOfClass:NSString.class] && [result length] ? result : @"Unknown";
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] opening lookup: %@", exception);
        return @"Unknown";
    }
}

static void CAPRememberFEN(NSString *fen) {
    if (!fen.length) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([currentFEN isEqualToString:fen]) return;
        currentFEN = [fen copy];
        openingName = CAPOpeningForFEN(fen);
        maiaMove = nil; stockfishMove = nil;
        hasMaiaWDL = NO; hasStockfishWDL = NO;
        lastAutoPlayedFEN = nil;
        CAPRefreshAll();
    });
}

static const OrtApi *CAPOrt(void) {
    static const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api;
}

static BOOL CAPOrtOK(OrtStatus *status, NSError **error) {
    if (!status) return YES;
    const OrtApi *api = CAPOrt();
    NSString *message = [NSString stringWithUTF8String:api->GetErrorMessage(status)];
    if (error) *error = [NSError errorWithDomain:@"ChessAssistantPro.ONNX" code:1
                                      userInfo:@{NSLocalizedDescriptionKey: message ?: @"ONNX Runtime error"}];
    api->ReleaseStatus(status);
    return NO;
}

static OrtSession *CAPMaiaSession(void) {
    static OrtEnv *env = nullptr;
    static OrtSession *session = nullptr;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (![[NSFileManager defaultManager] fileExistsAtPath:kModelPath]) return;
        const OrtApi *api = CAPOrt();
        if (!api) return;
        OrtSessionOptions *options = nullptr;
        if (!CAPOrtOK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ChessAssistantPro", &env), nullptr) ||
            !CAPOrtOK(api->CreateSessionOptions(&options), nullptr)) return;
        CAPOrtOK(api->SetIntraOpNumThreads(options, 2), nullptr);
        CAPOrtOK(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), nullptr);
        NSError *error = nil;
        if (!CAPOrtOK(api->CreateSession(env, kModelPath.fileSystemRepresentation, options, &session), &error))
            NSLog(@"[ChessAssistantPro] Maia ONNX load failed: %@", error);
        api->ReleaseSessionOptions(options);
    });
    return session;
}

static BOOL CAPInput(id<MLFeatureProvider> provider, NSString *name,
                     MLMultiArrayDataType type, NSArray<NSNumber *> *shape,
                     size_t bytes, void *destination) {
    MLMultiArray *array = [[provider featureValueForName:name] multiArrayValue];
    if (!array || array.dataType != type || ![array.shape isEqualToArray:shape] ||
        array.count * (type == MLMultiArrayDataTypeFloat32 ? sizeof(float) : sizeof(int32_t)) != bytes)
        return NO;
    // The original model's arrays are contiguous. Reject unusual strides.
    NSInteger stride = 1;
    for (NSInteger i = (NSInteger)shape.count - 1; i >= 0; --i) {
        if (array.strides[i].integerValue != stride) return NO;
        stride *= shape[i].integerValue;
    }
    memcpy(destination, array.dataPointer, bytes);
    return YES;
}

static id<MLFeatureProvider> CAPONNXPredict(id<MLFeatureProvider> input, NSError **error) {
    OrtSession *session = CAPMaiaSession();
    if (!session) {
        if (error) *error = [NSError errorWithDomain:@"ChessAssistantPro.ONNX" code:2
                                           userInfo:@{NSLocalizedDescriptionKey: @"Maia ONNX model unavailable"}];
        return nil;
    }
    float tokens[64 * 96];
    int32_t selfElo = 0, oppoElo = 0;
    if (!CAPInput(input, @"tokens", MLMultiArrayDataTypeFloat32, @[@1,@64,@96], sizeof(tokens), tokens) ||
        !CAPInput(input, @"self_elo", MLMultiArrayDataTypeInt32, @[@1], sizeof(selfElo), &selfElo) ||
        !CAPInput(input, @"oppo_elo", MLMultiArrayDataTypeInt32, @[@1], sizeof(oppoElo), &oppoElo)) {
        if (error) *error = [NSError errorWithDomain:@"ChessAssistantPro.ONNX" code:3
                                           userInfo:@{NSLocalizedDescriptionKey: @"Maia input shape or type mismatch"}];
        return nil;
    }
    const OrtApi *api = CAPOrt();
    OrtMemoryInfo *memory = nullptr;
    OrtValue *values[3] = {nullptr, nullptr, nullptr};
    OrtValue *outputs[2] = {nullptr, nullptr};
    const char *inputNames[] = {"tokens", "self_elo", "oppo_elo"};
    const char *outputNames[] = {"logits_move", "logits_value"};
    const int64_t tokenShape[] = {1,64,96};
    const int64_t eloShape[] = {1};
    id<MLFeatureProvider> result = nil;
    if (!CAPOrtOK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory), error)) goto cleanup;
    if (!CAPOrtOK(api->CreateTensorWithDataAsOrtValue(memory, tokens, sizeof(tokens), tokenShape, 3,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &values[0]), error)) goto cleanup;
    if (!CAPOrtOK(api->CreateTensorWithDataAsOrtValue(memory, &selfElo, sizeof(selfElo), eloShape, 1,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &values[1]), error)) goto cleanup;
    if (!CAPOrtOK(api->CreateTensorWithDataAsOrtValue(memory, &oppoElo, sizeof(oppoElo), eloShape, 1,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &values[2]), error)) goto cleanup;
    if (!CAPOrtOK(api->Run(session, nullptr, inputNames, (const OrtValue *const *)values, 3,
                           outputNames, 2, outputs), error)) goto cleanup;
    {
        const size_t counts[] = {4352,3};
        const char *names[] = {"move_logits", "value_logits"};
        NSMutableDictionary *features = [NSMutableDictionary dictionaryWithCapacity:2];
        for (int i = 0; i < 2; ++i) {
            OrtTensorTypeAndShapeInfo *info = nullptr;
            void *data = nullptr;
            size_t count = 0;
            ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            BOOL valid = CAPOrtOK(api->GetTensorTypeAndShape(outputs[i], &info), error) &&
                         CAPOrtOK(api->GetTensorShapeElementCount(info, &count), error) &&
                         CAPOrtOK(api->GetTensorElementType(info, &elementType), error) &&
                         elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                         count == counts[i] &&
                         CAPOrtOK(api->GetTensorMutableData(outputs[i], &data), error);
            if (info) api->ReleaseTensorTypeAndShapeInfo(info);
            if (!valid) goto cleanup;
            MLMultiArray *array = [[MLMultiArray alloc] initWithShape:@[@1,@(counts[i])]
                                                             dataType:MLMultiArrayDataTypeFloat32 error:error];
            if (!array) goto cleanup;
            memcpy(array.dataPointer, data, count * sizeof(float));
            features[[NSString stringWithUTF8String:names[i]]] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        result = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:error];
    }
cleanup:
    for (OrtValue *value : outputs) if (value) api->ReleaseValue(value);
    for (OrtValue *value : values) if (value) api->ReleaseValue(value);
    if (memory) api->ReleaseMemoryInfo(memory);
    return result;
}

// The original MaiaGo routine performs FEN -> token encoding and logits ->
// legal move decoding. Replace only Core ML prediction to preserve that map.
typedef id<MLFeatureProvider> (*PredictIMP)(MLModel *, SEL, id<MLFeatureProvider>, NSError **);
static PredictIMP originalPredict = nullptr;
static id<MLFeatureProvider> CAPModifiedPredict(MLModel *self, SEL selector,
                                                id<MLFeatureProvider> input, NSError **error) {
    @try {
        if (CAPMaiaEnabled() &&
            [input.featureNames containsObject:@"tokens"] &&
            [input.featureNames containsObject:@"self_elo"] &&
            [input.featureNames containsObject:@"oppo_elo"]) {
            id<MLFeatureProvider> output = CAPONNXPredict(input, error);
            MLMultiArray *logits = [[output featureValueForName:@"value_logits"] multiArrayValue];
            if (logits.count == 3 && logits.dataType == MLMultiArrayDataTypeFloat32) {
                const float *values = (const float *)logits.dataPointer;
                double highest = MAX(values[0], MAX(values[1], values[2]));
                double weights[3] = {exp(values[0] - highest), exp(values[1] - highest), exp(values[2] - highest)};
                double total = weights[0] + weights[1] + weights[2];
                double lossWeight = weights[0], drawWeight = weights[1], winWeight = weights[2];
                [CAPFenLock() lock];
                NSString *fen = [predictionFEN copy];
                [CAPFenLock() unlock];
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (![currentFEN isEqualToString:fen] || total <= 0) return;
                    BOOL whiteToMove = [fen containsString:@" w "];
                    // Maia3 value labels are loss, draw, win for the side to move.
                    maiaWDL[0] = (whiteToMove ? winWeight : lossWeight) / total;
                    maiaWDL[1] = drawWeight / total;
                    maiaWDL[2] = (whiteToMove ? lossWeight : winWeight) / total;
                    hasMaiaWDL = YES;
                    CAPRefreshAll();
                });
            }
            return output;
        }
        return originalPredict(self, selector, input, error);
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] ONNX prediction: %@", exception);
        if (error) *error = [NSError errorWithDomain:@"ChessAssistantPro" code:2 userInfo:nil];
        return nil;
    }
}

static void CAPEmptyMaia(MaiaCallback callback) {
    if (!callback) return;
    MaiaCallback copied = [callback copy];
    dispatch_async(dispatch_get_main_queue(), ^{ MaiaResult result = {}; copied(result); });
}

static void CAPMaiaGo(const char *fen, uintptr_t selfElo, uintptr_t oppoElo,
                      MaiaCallback callback) {
    @try {
        size_t length = fen ? strnlen(fen, 256) : 0;
        if (!CAPMaiaEnabled() || length == 0 || length >= 256 || !CAPMaiaSession() || !originalMaiaGo) {
            CAPEmptyMaia(callback);
            return;
        }
        NSString *position = [[NSString alloc] initWithBytes:fen length:length encoding:NSUTF8StringEncoding];
        if (!position) { CAPEmptyMaia(callback); return; }
        [CAPFenLock() lock]; predictionFEN = [position copy]; [CAPFenLock() unlock];
        CAPRememberFEN(position);
        NSInteger elo = CAPMaiaElo();
        MaiaCallback forwarded = ^(MaiaResult result) {
            size_t moveLength = strnlen(result.move, sizeof(result.move));
            if (moveLength >= 4 && moveLength <= 5) {
                NSString *move = [[NSString alloc] initWithBytes:result.move length:moveLength encoding:NSASCIIStringEncoding];
                CAPRememberMove(NO, move, position);
            }
            if (callback) callback(result);
        };
        originalMaiaGo(fen, (uintptr_t)elo, (uintptr_t)elo, [forwarded copy]);
        if (CAPStockfishEnabled()) CAPStartStockfish(fen, 20, 1, 20, nil);
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] MaiaGo: %@", exception);
        CAPEmptyMaia(callback);
    }
    (void)selfElo; (void)oppoElo;
}

static void CAPEmptyEngine(EngineCallback callback) {
    if (!callback) return;
    EngineCallback copied = [callback copy];
    dispatch_async(dispatch_get_main_queue(), ^{ copied(nullptr, 0); });
}

static void CAPStartStockfish(const char *fen, uintptr_t arg1, uintptr_t arg2,
                              uintptr_t arg3, EngineCallback callback) {
    size_t length = fen ? strnlen(fen, 256) : 0;
    if (!CAPStockfishEnabled() || length == 0 || length >= 256 ||
        strpbrk(fen, "\r\n") || !originalEngineGo) {
        CAPEmptyEngine(callback);
        return;
    }
    NSString *position = [[NSString alloc] initWithBytes:fen length:length
                                                encoding:NSUTF8StringEncoding];
    if (!position) { CAPEmptyEngine(callback); return; }
    CAPRememberFEN(position);
    EngineCallback forwarded = ^(const EngineLine *lines, int count) {
        if (lines && count > 0) {
            size_t moveLength = strnlen(lines[0].move, sizeof(lines[0].move));
            if (moveLength >= 4 && moveLength <= 5) {
                NSString *move = [[NSString alloc] initWithBytes:lines[0].move
                                                          length:moveLength encoding:NSASCIIStringEncoding];
                CAPRememberMove(YES, move, position);
            }
        }
        if (callback) callback(lines, count);
    };
    // EngineGo is the embedded Stockfish 16 entry point. Keep its ABI intact.
    originalEngineGo(fen, arg1, arg2, arg3, [forwarded copy]);
}

static void CAPEngineGo(const char *fen, uintptr_t arg1, uintptr_t arg2,
                        uintptr_t arg3, EngineCallback callback) {
    @try { CAPStartStockfish(fen, arg1, arg2, arg3, callback); }
    @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] EngineGo: %@", exception);
        CAPEmptyEngine(callback);
    }
}

static UIView *CAPFindBoardIn(UIView *root) {
    Class boardClass = NSClassFromString(@"CHBoardView");
    if (boardClass && [root isKindOfClass:boardClass] && !root.hidden && root.bounds.size.width > 160) return root;
    for (UIView *child in root.subviews) {
        UIView *found = CAPFindBoardIn(child);
        if (found) return found;
    }
    return nil;
}

static UIView *CAPVisibleBoard(void) {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) continue;
        for (UIWindow *window in [((UIWindowScene *)scene).windows reverseObjectEnumerator]) {
            UIView *found = CAPFindBoardIn(window);
            if (found) return found;
        }
    }
    return nil;
}

static NSString *CAPNormalizedMove(NSString *move) {
    NSMutableString *result = [NSMutableString string];
    NSCharacterSet *allowed = NSCharacterSet.alphanumericCharacterSet;
    for (NSUInteger i = 0; i < move.length; ++i) {
        unichar c = [move characterAtIndex:i];
        if ([allowed characterIsMember:c]) [result appendFormat:@"%C", c];
    }
    return result.lowercaseString;
}

static void CAPTryAutoPlay(BOOL fromStockfish, NSString *move, NSString *fen) {
    BOOL chosen = fromStockfish ? (CAPStockfishEnabled() && CAPStockfishAuto())
                                : (CAPMaiaEnabled() && CAPMaiaAuto() &&
                                   !(CAPStockfishEnabled() && CAPStockfishAuto()));
    if (!chosen || ![currentFEN isEqualToString:fen] || [lastAutoPlayedFEN isEqualToString:fen] || move.length < 4) return;
    @try {
        UIView *board = CAPVisibleBoard();
        if (!board || ![board respondsToSelector:@selector(makeMove:throughTouches:)] ||
            ![board respondsToSelector:@selector(chessGame)]) return;
        BOOL whiteToMove = [fen containsString:@" w "];
        SEL allowedSelector = whiteToMove ? @selector(allowUserToMakeWhiteMoves) : @selector(allowUserToMakeBlackMoves);
        if ([board respondsToSelector:allowedSelector] &&
            !((BOOL (*)(id, SEL))objc_msgSend)(board, allowedSelector)) return;
        id game = ((id (*)(id, SEL))objc_msgSend)(board, @selector(chessGame));
        if (!game || ![game respondsToSelector:@selector(isValidMoveFromSourceRank:SourceFile:DestRank:DestFile:)]) return;
        Class moveClass = NSClassFromString(@"CHMoveSourceDestination");
        if (!moveClass) return;
        NSString *wanted = CAPNormalizedMove(move);
        int sf = [[wanted substringWithRange:NSMakeRange(0, 1)] characterAtIndex:0] - 'a';
        int sr = [[wanted substringWithRange:NSMakeRange(1, 1)] characterAtIndex:0] - '1';
        int df = [[wanted substringWithRange:NSMakeRange(2, 1)] characterAtIndex:0] - 'a';
        int dr = [[wanted substringWithRange:NSMakeRange(3, 1)] characterAtIndex:0] - '1';
        if (sf < 0 || sf > 7 || df < 0 || df > 7 || sr < 0 || sr > 7 || dr < 0 || dr > 7) return;
        char promotion = wanted.length == 5 ? (char)[wanted characterAtIndex:4] : 0;
        for (int base = 0; base <= 1; ++base) {
            char sourceRank = (char)(sr + base), sourceFile = (char)(sf + base);
            char destRank = (char)(dr + base), destFile = (char)(df + base);
            id candidate = nil;
            SEL initSelector = @selector(initSR:SF:DR:DF:PP:);
            candidate = ((id (*)(id, SEL, char, char, char, char, char))objc_msgSend)
                ([moveClass alloc], initSelector, sourceRank, sourceFile, destRank, destFile, promotion);
            NSString *lan = [candidate respondsToSelector:@selector(lanMove)] ?
                ((id (*)(id, SEL))objc_msgSend)(candidate, @selector(lanMove)) : nil;
            if (![CAPNormalizedMove(lan ?: @"") isEqualToString:wanted]) continue;
            BOOL legal = ((BOOL (*)(id, SEL, char, char, char, char))objc_msgSend)
                (game, @selector(isValidMoveFromSourceRank:SourceFile:DestRank:DestFile:),
                 sourceRank, sourceFile, destRank, destFile);
            if (!legal) return;
            lastAutoPlayedFEN = [fen copy];
            ((void (*)(id, SEL, id, BOOL))objc_msgSend)(board, @selector(makeMove:throughTouches:), candidate, YES);
            return;
        }
    } @catch (NSException *exception) {
        lastAutoPlayedFEN = nil;
        NSLog(@"[ChessAssistantPro] Auto-Play skipped: %@", exception);
    }
}

static CGPoint CAPSquarePoint(int file, int rank, CGRect bounds, BOOL flipped) {
    CGFloat side = MIN(bounds.size.width, bounds.size.height);
    CGFloat unit = side / 8.0;
    CGFloat left = bounds.origin.x + (bounds.size.width - side) / 2.0;
    CGFloat top = bounds.origin.y + (bounds.size.height - side) / 2.0;
    int column = flipped ? 7 - file : file;
    int row = flipped ? rank : 7 - rank;
    return CGPointMake(left + (column + 0.5) * unit, top + (row + 0.5) * unit);
}

static void CAPAddArrow(UIView *overlay, NSString *move, UIColor *color, BOOL flipped) {
    NSString *uci = CAPNormalizedMove(move ?: @"");
    if (uci.length < 4) return;
    int sf = [uci characterAtIndex:0] - 'a', sr = [uci characterAtIndex:1] - '1';
    int df = [uci characterAtIndex:2] - 'a', dr = [uci characterAtIndex:3] - '1';
    if (sf < 0 || sf > 7 || df < 0 || df > 7 || sr < 0 || sr > 7 || dr < 0 || dr > 7) return;
    CGPoint start = CAPSquarePoint(sf, sr, overlay.bounds, flipped);
    CGPoint end = CAPSquarePoint(df, dr, overlay.bounds, flipped);
    CGFloat dx = end.x - start.x, dy = end.y - start.y;
    CGFloat length = hypot(dx, dy);
    if (length < 1) return;
    CGFloat ux = dx / length, uy = dy / length;
    CGFloat width = MIN(overlay.bounds.size.width, overlay.bounds.size.height) / 8.0;
    CGFloat head = width * 0.28;
    CGPoint shaftEnd = CGPointMake(end.x - ux * head * 0.7, end.y - uy * head * 0.7);
    CGPoint wing1 = CGPointMake(end.x - ux * head - uy * head * 0.55, end.y - uy * head + ux * head * 0.55);
    CGPoint wing2 = CGPointMake(end.x - ux * head + uy * head * 0.55, end.y - uy * head - ux * head * 0.55);
    UIBezierPath *path = [UIBezierPath bezierPath];
    [path moveToPoint:start]; [path addLineToPoint:shaftEnd];
    [path moveToPoint:end]; [path addLineToPoint:wing1]; [path addLineToPoint:wing2]; [path closePath];
    CAShapeLayer *arrow = [CAShapeLayer layer];
    arrow.path = path.CGPath;
    arrow.strokeColor = color.CGColor;
    arrow.fillColor = color.CGColor;
    arrow.lineWidth = width * 0.12;
    arrow.lineCap = kCALineCapRound;
    arrow.lineJoin = kCALineJoinRound;
    arrow.frame = overlay.bounds;
    objc_setAssociatedObject(arrow, &kArrowLayerKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [overlay.layer addSublayer:arrow];
}

static void CAPDrawArrows(void) {
    UIView *board = CAPVisibleBoard();
    if (!board) return;
    UIView *overlay = [board respondsToSelector:@selector(boardArrowsView)] ?
        ((id (*)(id, SEL))objc_msgSend)(board, @selector(boardArrowsView)) : board;
    if (![overlay isKindOfClass:UIView.class]) overlay = board;
    CAPRemoveMarkedLayers(overlay.layer);
    BOOL flipped = [board respondsToSelector:@selector(isFlipped)] &&
        ((BOOL (*)(id, SEL))objc_msgSend)(board, @selector(isFlipped));
    if (CAPMaiaEnabled()) CAPAddArrow(overlay, maiaMove, [UIColor colorWithRed:0.22 green:0.88 blue:0.52 alpha:0.9], flipped);
    if (CAPStockfishEnabled()) CAPAddArrow(overlay, stockfishMove, [UIColor colorWithRed:0.25 green:0.55 blue:1 alpha:0.9], flipped);
}

static void CAPRememberMove(BOOL fromStockfish, NSString *move, NSString *fen) {
    if (!move.length || !fen.length) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (![currentFEN isEqualToString:fen]) return;
        if (fromStockfish) stockfishMove = [move copy]; else maiaMove = [move copy];
        CAPDrawArrows();
        CAPTryAutoPlay(fromStockfish, move, fen);
        CAPRefreshAll();
    });
}

@interface CAPWinBar : UIView
@property(nonatomic) double whiteChance;
@property(nonatomic) double drawChance;
@property(nonatomic) double blackChance;
@end
@implementation CAPWinBar
- (void)drawRect:(CGRect)rect {
    CGFloat width = rect.size.width;
    if (self.whiteChance + self.drawChance + self.blackChance <= 0) {
        [[UIColor colorWithWhite:0.28 alpha:1] setFill]; UIRectFill(rect); return;
    }
    CGFloat white = MAX(0, MIN(1, self.whiteChance));
    CGFloat draw = MAX(0, MIN(1 - white, self.drawChance));
    [[UIColor colorWithWhite:0.93 alpha:1] setFill]; UIRectFill(CGRectMake(0, 0, width * white, rect.size.height));
    [[UIColor colorWithWhite:0.53 alpha:1] setFill]; UIRectFill(CGRectMake(width * white, 0, width * draw, rect.size.height));
    [[UIColor colorWithWhite:0.15 alpha:1] setFill]; UIRectFill(CGRectMake(width * (white + draw), 0, width * (1 - white - draw), rect.size.height));
}
@end

static UILabel *CAPLabel(NSString *text, CGFloat size, BOOL bold) {
    UILabel *label = [UILabel new];
    label.text = text;
    label.textColor = UIColor.whiteColor;
    label.font = bold ? [UIFont boldSystemFontOfSize:size] : [UIFont systemFontOfSize:size];
    label.numberOfLines = 0;
    return label;
}

static UIStackView *CAPStack(NSArray<UIView *> *items, CGFloat spacing) {
    UIStackView *stack = [[UIStackView alloc] initWithArrangedSubviews:items];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = spacing;
    return stack;
}

static UIButton *CAPButton(NSString *text, id target, SEL action, NSInteger tag) {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    button.tag = tag;
    button.backgroundColor = [UIColor colorWithWhite:0.16 alpha:1];
    button.layer.cornerRadius = 12;
    button.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeft;
    button.titleLabel.numberOfLines = 2;
    button.titleLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightSemibold];
    [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    [button setTitle:text forState:UIControlStateNormal];
    [button addTarget:target action:action forControlEvents:UIControlEventTouchUpInside];
    [button.heightAnchor constraintGreaterThanOrEqualToConstant:64].active = YES;
    return button;
}

static UIView *CAPToggleRow(NSString *text, UISwitch *toggle) {
    UILabel *label = CAPLabel(text, 16, NO);
    [label setContentCompressionResistancePriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    UIStackView *row = [[UIStackView alloc] initWithArrangedSubviews:@[label, toggle]];
    row.axis = UILayoutConstraintAxisHorizontal;
    row.alignment = UIStackViewAlignmentCenter;
    row.distribution = UIStackViewDistributionEqualSpacing;
    [row.heightAnchor constraintGreaterThanOrEqualToConstant:48].active = YES;
    return row;
}

static UISwitch *CAPNativeSwitch(UIView *panel, BOOL on, SEL action, NSInteger tag) {
    id value = ((id (*)(id, SEL, BOOL, SEL))objc_msgSend)(panel, @selector(switchOn:sel:), on, action);
    UISwitch *toggle = [value isKindOfClass:UISwitch.class] ? value : [UISwitch new];
    [toggle removeTarget:nil action:NULL forControlEvents:UIControlEventValueChanged];
    [toggle addTarget:panel action:action forControlEvents:UIControlEventValueChanged];
    toggle.onTintColor = [UIColor colorWithRed:0.24 green:0.82 blue:0.65 alpha:1];
    toggle.tag = tag;
    return toggle;
}

static NSString *CAPMaiaAdvice(void) {
    NSInteger elo = CAPMaiaElo();
    return elo < 1800 ? @"Safe for Blitz/Bullet" : elo < 2300 ? @"Moderate Risk" : @"High Risk / Perfect Play";
}
static NSString *CAPStockfishAdvice(void) {
    NSInteger elo = CAPStockfishStrength();
    return elo < 2000 ? @"Safe for Bots/Beginners" : elo < 2800 ? @"Moderate Risk against Humans" : @"Extreme Risk / Superhuman";
}

static void CAPRefreshPanel(UIView *panel) {
    UIView *container = objc_getAssociatedObject(panel, &kPanelKey);
    if (!container) return;
    NSInteger screen = [objc_getAssociatedObject(panel, &kScreenKey) integerValue];
    [container viewWithTag:9001].hidden = screen != 0;
    [container viewWithTag:9002].hidden = screen != 1;
    [container viewWithTag:9003].hidden = screen != 2;
    [(UIButton *)[container viewWithTag:9011] setTitle:
        [NSString stringWithFormat:@"Maia Settings\nMaia: %@ (%ld ELO)", CAPMaiaEnabled() ? @"ON" : @"OFF", (long)CAPMaiaElo()]
        forState:UIControlStateNormal];
    [(UIButton *)[container viewWithTag:9012] setTitle:
        [NSString stringWithFormat:@"Stockfish Settings\nStockfish: %@ (Depth %d)", CAPStockfishEnabled() ? @"ON" : @"OFF", CAPStockfishDepth()]
        forState:UIControlStateNormal];
    ((UISwitch *)[container viewWithTag:9021]).on = CAPMaiaEnabled();
    ((UISwitch *)[container viewWithTag:9031]).on = CAPStockfishEnabled();
    ((UISlider *)[container viewWithTag:9022]).value = (float)CAPMaiaElo();
    ((UISlider *)[container viewWithTag:9032]).value = (float)CAPStockfishStrength();
    ((UISegmentedControl *)[container viewWithTag:9024]).selectedSegmentIndex = CAPMaiaAuto() ? 1 : 0;
    ((UISegmentedControl *)[container viewWithTag:9034]).selectedSegmentIndex = CAPStockfishAuto() ? 1 : 0;
    ((UILabel *)[container viewWithTag:9023]).text = [NSString stringWithFormat:@"Maia ELO: %ld\n%@", (long)CAPMaiaElo(), CAPMaiaAdvice()];
    NSInteger strength = CAPStockfishStrength();
    NSString *rating = strength >= 3200 ? @"Maximum" : [NSString stringWithFormat:@"%ld ELO", (long)strength];
    ((UILabel *)[container viewWithTag:9033]).text = [NSString stringWithFormat:@"Stockfish Strength: %@ · Depth %d\n%@", rating, CAPStockfishDepth(), CAPStockfishAdvice()];
    ((UILabel *)[container viewWithTag:9013]).text = [NSString stringWithFormat:@"Opening: %@", openingName ?: @"—"];
    BOOL showSF = CAPStockfishEnabled() && hasStockfishWDL;
    BOOL showMaia = !showSF && CAPMaiaEnabled() && hasMaiaWDL;
    double white = showSF ? stockfishWDL[0] : showMaia ? maiaWDL[0] : 0;
    double draw = showSF ? stockfishWDL[1] : showMaia ? maiaWDL[1] : 0;
    double black = showSF ? stockfishWDL[2] : showMaia ? maiaWDL[2] : 0;
    ((UILabel *)[container viewWithTag:9014]).text = showSF ? @"Win probabilities · Stockfish" : showMaia ? @"Win probabilities · Maia" : @"Win probabilities · awaiting analysis";
    ((UILabel *)[container viewWithTag:9015]).text = showSF || showMaia ?
        [NSString stringWithFormat:@"White %.0f%%   Draw %.0f%%   Black %.0f%%", white * 100, draw * 100, black * 100] :
        @"White —   Draw —   Black —";
    CAPWinBar *bar = (CAPWinBar *)[container viewWithTag:9016];
    bar.whiteChance = white; bar.drawChance = draw; bar.blackChance = black;
    [bar setNeedsDisplay];
}

static void CAPRefreshAll(void) {
    if (![NSThread isMainThread]) { dispatch_async(dispatch_get_main_queue(), ^{ CAPRefreshAll(); }); return; }
    for (UIView *panel in visiblePanels.allObjects) CAPRefreshPanel(panel);
    CAPDrawArrows();
}

static void CAPInstallPanel(UIView *panel) {
    UIView *container = objc_getAssociatedObject(panel, &kPanelKey);
    if (!container) {
        if (!visiblePanels) visiblePanels = [NSHashTable weakObjectsHashTable];
        [visiblePanels addObject:panel];
        objc_setAssociatedObject(panel, &kScreenKey, @0, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        container = [UIView new];
        container.backgroundColor = [UIColor colorWithRed:0.07 green:0.09 blue:0.13 alpha:1];
        container.overrideUserInterfaceStyle = UIUserInterfaceStyleDark;
        container.translatesAutoresizingMaskIntoConstraints = NO;
        objc_setAssociatedObject(panel, &kPanelKey, container, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [panel addSubview:container];
        UIScrollView *scroll = [UIScrollView new];
        scroll.translatesAutoresizingMaskIntoConstraints = NO;
        scroll.alwaysBounceVertical = YES;
        [container addSubview:scroll];
        UILabel *header = CAPLabel(@"Chess Assistant Pro", 22, YES);
        UIButton *close = [UIButton buttonWithType:UIButtonTypeSystem];
        [close setTitle:@"Close" forState:UIControlStateNormal];
        [close addTarget:panel action:@selector(closeTap) forControlEvents:UIControlEventTouchUpInside];
        UIStackView *heading = [[UIStackView alloc] initWithArrangedSubviews:@[header, close]];
        heading.axis = UILayoutConstraintAxisHorizontal;
        heading.alignment = UIStackViewAlignmentCenter;
        heading.distribution = UIStackViewDistributionEqualSpacing;

        UILabel *opening = CAPLabel(@"Opening: —", 15, NO); opening.tag = 9013;
        UILabel *wdlTitle = CAPLabel(@"Win probabilities · awaiting analysis", 14, YES); wdlTitle.tag = 9014;
        CAPWinBar *bar = [CAPWinBar new]; bar.tag = 9016;
        bar.backgroundColor = [UIColor colorWithWhite:0.18 alpha:1];
        bar.layer.cornerRadius = 7; bar.clipsToBounds = YES;
        [bar.heightAnchor constraintEqualToConstant:14].active = YES;
        UILabel *numbers = CAPLabel(@"White —   Draw —   Black —", 13, NO); numbers.tag = 9015;
        UIStackView *main = CAPStack(@[
            CAPButton(@"Maia Settings", panel, @selector(cap_openMaia), 9011),
            CAPButton(@"Stockfish Settings", panel, @selector(cap_openStockfish), 9012),
            opening, wdlTitle, bar, numbers
        ], 14); main.tag = 9001;

        UISwitch *maiaToggle = CAPNativeSwitch(panel, CAPMaiaEnabled(), @selector(cap_maiaChanged:), 9021);
        UISlider *maiaSlider = [UISlider new]; maiaSlider.tag = 9022;
        maiaSlider.minimumValue = 1100; maiaSlider.maximumValue = 2700;
        [maiaSlider addTarget:panel action:@selector(cap_maiaEloChanged:) forControlEvents:UIControlEventValueChanged];
        UILabel *maiaInfo = CAPLabel(@"", 15, NO); maiaInfo.tag = 9023;
        UISegmentedControl *maiaMode = [[UISegmentedControl alloc] initWithItems:@[@"Suggest Only", @"Auto-Play"]];
        maiaMode.tag = 9024; [maiaMode addTarget:panel action:@selector(cap_maiaPlayMode:) forControlEvents:UIControlEventValueChanged];
        UIStackView *maia = CAPStack(@[
            CAPButton(@"‹  Back", panel, @selector(cap_back), 0), CAPLabel(@"Maia Settings", 19, YES),
            CAPToggleRow(@"Enable Maia", maiaToggle), maiaInfo, maiaSlider,
            CAPLabel(@"Play Mode", 14, YES), maiaMode
        ], 15); maia.tag = 9002;

        UISwitch *stockfishToggle = CAPNativeSwitch(panel, CAPStockfishEnabled(), @selector(cap_stockfishChanged:), 9031);
        UISlider *stockfishSlider = [UISlider new]; stockfishSlider.tag = 9032;
        stockfishSlider.minimumValue = 1320; stockfishSlider.maximumValue = 3200;
        [stockfishSlider addTarget:panel action:@selector(cap_stockfishStrengthChanged:) forControlEvents:UIControlEventValueChanged];
        UILabel *stockfishInfo = CAPLabel(@"", 15, NO); stockfishInfo.tag = 9033;
        UISegmentedControl *stockfishMode = [[UISegmentedControl alloc] initWithItems:@[@"Suggest Only", @"Auto-Play"]];
        stockfishMode.tag = 9034; [stockfishMode addTarget:panel action:@selector(cap_stockfishPlayMode:) forControlEvents:UIControlEventValueChanged];
        UIStackView *stockfish = CAPStack(@[
            CAPButton(@"‹  Back", panel, @selector(cap_back), 0), CAPLabel(@"Stockfish Settings", 19, YES),
            CAPToggleRow(@"Enable Stockfish", stockfishToggle), stockfishInfo, stockfishSlider,
            CAPLabel(@"Play Mode", 14, YES), stockfishMode
        ], 15); stockfish.tag = 9003;
        UIStackView *root = CAPStack(@[heading, main, maia, stockfish], 20);
        root.translatesAutoresizingMaskIntoConstraints = NO;
        [scroll addSubview:root];
        [NSLayoutConstraint activateConstraints:@[
            [container.topAnchor constraintEqualToAnchor:panel.topAnchor],
            [container.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor],
            [container.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor],
            [container.bottomAnchor constraintEqualToAnchor:panel.bottomAnchor],
            [scroll.topAnchor constraintEqualToAnchor:container.topAnchor],
            [scroll.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
            [scroll.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
            [scroll.bottomAnchor constraintEqualToAnchor:container.bottomAnchor],
            [root.topAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.topAnchor constant:20],
            [root.leadingAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.leadingAnchor constant:20],
            [root.trailingAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.trailingAnchor constant:-20],
            [root.bottomAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.bottomAnchor constant:-20],
            [root.widthAnchor constraintEqualToAnchor:scroll.frameLayoutGuide.widthAnchor constant:-40]
        ]];
    }
    for (UIView *child in panel.subviews) child.hidden = child != container;
    container.hidden = NO;
    [panel bringSubviewToFront:container];
    CAPRefreshPanel(panel);
}

%hook CHSettingsPanel
- (void)build { %orig; CAPInstallPanel((UIView *)self); }
- (void)populate { %orig; CAPInstallPanel((UIView *)self); }
%new
- (void)cap_openMaia { objc_setAssociatedObject(self, &kScreenKey, @1, OBJC_ASSOCIATION_RETAIN_NONATOMIC); CAPRefreshPanel((UIView *)self); }
%new
- (void)cap_openStockfish { objc_setAssociatedObject(self, &kScreenKey, @2, OBJC_ASSOCIATION_RETAIN_NONATOMIC); CAPRefreshPanel((UIView *)self); }
%new
- (void)cap_back { objc_setAssociatedObject(self, &kScreenKey, @0, OBJC_ASSOCIATION_RETAIN_NONATOMIC); CAPRefreshPanel((UIView *)self); }
%new
- (void)cap_maiaChanged:(UISwitch *)sender { CAPSetMode(sender.on, CAPStockfishEnabled()); }
%new
- (void)cap_stockfishChanged:(UISwitch *)sender { CAPSetMode(CAPMaiaEnabled(), sender.on); }
%new
- (void)cap_maiaEloChanged:(UISlider *)slider {
    NSInteger value = (NSInteger)(roundf(slider.value / 50) * 50);
    [[NSUserDefaults standardUserDefaults] setInteger:value forKey:kMaiaEloKey];
    CAPRefreshAll();
}
%new
- (void)cap_stockfishStrengthChanged:(UISlider *)slider {
    NSInteger value = slider.value >= 3180 ? 3200 : (NSInteger)(roundf(slider.value / 20) * 20);
    [[NSUserDefaults standardUserDefaults] setInteger:value forKey:kStockfishStrengthKey];
    CAPRefreshAll();
}
%new
- (void)cap_maiaPlayMode:(UISegmentedControl *)control {
    [[NSUserDefaults standardUserDefaults] setBool:control.selectedSegmentIndex == 1 forKey:kMaiaAutoKey];
    CAPRefreshAll();
}
%new
- (void)cap_stockfishPlayMode:(UISegmentedControl *)control {
    [[NSUserDefaults standardUserDefaults] setBool:control.selectedSegmentIndex == 1 forKey:kStockfishAutoKey];
    CAPRefreshAll();
}
%end

%hook CHBoardView
- (void)layoutSubviews {
    %orig;
    if ([NSThread isMainThread]) CAPDrawArrows();
}
%end

%hook CALayer
- (void)addSublayer:(CALayer *)layer {
    Dl_info caller = {};
    void *returnAddress = __builtin_return_address(0);
    BOOL fromAssistant = dladdr(returnAddress, &caller) != 0 && caller.dli_fname &&
                         strstr(caller.dli_fname, "/Chess.dylib") != nullptr;
    if (fromAssistant && [layer isKindOfClass:CAShapeLayer.class]) {
        // The original tweak's arrow layers are replaced with our color-coded layers.
        return;
    }
    %orig;
}
%end

%ctor {
    @autoreleasepool {
        %init;
        Class modelClass = objc_getClass("MLModel");
        if (modelClass) MSHookMessageEx(modelClass, @selector(predictionFromFeatures:error:),
                                        (IMP)CAPModifiedPredict, (IMP *)&originalPredict);
        void *engine = dlsym(RTLD_DEFAULT, "EngineGo");
        void *maia = dlsym(RTLD_DEFAULT, "MaiaGo");
        if (engine) MSHookFunction(engine, (void *)CAPEngineGo, (void **)&originalEngineGo);
        if (maia) MSHookFunction(maia, (void *)CAPMaiaGo, (void **)&originalMaiaGo);
        NSLog(@"[ChessAssistantPro] hooks engine=%p maia=%p", engine, maia);
    }
}
