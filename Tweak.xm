#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <substrate.h>
#import <dlfcn.h>
#import <dispatch/dispatch.h>
#import <QuartzCore/QuartzCore.h>
#import <mach-o/loader.h>
#import <mach-o/nlist.h>
#import <mach-o/dyld.h>
#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <atomic>
#include <sstream>
#include <string>

extern char **environ;

static NSString *const kModelPath = @"/var/jb/Library/Application Support/ChessAssistantPro/maia3_79m_int32.onnx";
static NSString *const kAutoPlayKey = @"ChessAssistantPro_AutoPlay";
static NSString *const kAutoPlayMinDelayKey = @"ChessAssistantPro_AutoPlayMinDelay";
static NSString *const kAutoPlayMaxDelayKey = @"ChessAssistantPro_AutoPlayMaxDelay";
static NSString *const kStrengthEloKey = @"ChessAssist_ELO";
static NSString *const kGodEngineKey = @"godEngineChoice";
static NSString *const kAutoEngineKey = @"ChessAssistantPro_AutoPlayEngine";
static const char *kStockfish19Path = "/var/jb/usr/bin/stockfish19";
static const char *kTriumviratusPath = "/var/jb/usr/bin/triumviratus";
static NSString *capLastOutgoingMove = nil;

static NSString *CAPDumpPath(void) {
    NSArray<NSString *> *paths = NSSearchPathForDirectoriesInDomains(
        NSDocumentDirectory, NSUserDomainMask, YES);
    return paths.firstObject ?
        [paths.firstObject stringByAppendingPathComponent:@"cap_swift_dump.txt"] : nil;
}

static void CAPDiscLog(NSString *format, ...) {
    va_list args;
    va_start(args, format);
    NSString *message = [[NSString alloc] initWithFormat:format arguments:args];
    va_end(args);
    NSLog(@"[CAP-DISC] %@", message);
    NSString *path = CAPDumpPath();
    if (!path) return;
    static dispatch_queue_t writeQueue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        writeQueue = dispatch_queue_create("com.jerich.chessassistantpro.discovery", DISPATCH_QUEUE_SERIAL);
    });
    NSString *line = [NSString stringWithFormat:@"[CAP-DISC] %@\n", message];
    dispatch_async(writeQueue, ^{
        [[NSFileManager defaultManager] createDirectoryAtPath:[path stringByDeletingLastPathComponent]
                                  withIntermediateDirectories:YES attributes:nil error:nil];
        const char *file = path.fileSystemRepresentation;
        int fd = open(file, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (fd < 0) return;
        NSData *bytes = [line dataUsingEncoding:NSUTF8StringEncoding];
        const uint8_t *cursor = (const uint8_t *)bytes.bytes;
        NSUInteger remaining = bytes.length;
        while (remaining) {
            ssize_t written = write(fd, cursor, remaining);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) break;
            cursor += written;
            remaining -= (NSUInteger)written;
        }
        close(fd);
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

// Keep the original Maia tokenization, legal-move decoder, callback and arrows.
// Only replace the prediction output consumed by that decoder.
typedef id<MLFeatureProvider> (*PredictIMP)(MLModel *, SEL, id<MLFeatureProvider>, NSError **);
static PredictIMP originalPredict = nullptr;

static id<MLFeatureProvider> CAPModifiedPredict(MLModel *model, SEL selector,
                                                id<MLFeatureProvider> input, NSError **error) {
    if (![input.featureNames containsObject:@"tokens"] ||
        ![input.featureNames containsObject:@"self_elo"] ||
        ![input.featureNames containsObject:@"oppo_elo"])
        return originalPredict(model, selector, input, error);
    @try {
        NSError *onnxError = nil;
        id<MLFeatureProvider> output = CAPONNXPredict(input, &onnxError);
        if (output &&
            [[[output featureValueForName:@"move_logits"] multiArrayValue].shape isEqualToArray:@[@1,@4352]] &&
            [[[output featureValueForName:@"value_logits"] multiArrayValue].shape isEqualToArray:@[@1,@3]])
            return output;
        NSLog(@"[ChessAssistantPro] Maia ONNX unavailable; using original model: %@", onnxError);
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] Maia ONNX exception: %@", exception);
    }
    return originalPredict(model, selector, input, error);
}

// The recovered block ABI and result layout are from Chess.dylib. The wrapper
// forwards the full result to the original CHPad logic without modification.
struct MaiaResult {
    char move[8]; double value1; double value2; bool flag; int value3;
    char moves[4][8]; double values[4];
};
static_assert(sizeof(MaiaResult) == 96, "Maia callback ABI changed");
typedef void (^MaiaCallback)(MaiaResult);
typedef void (*MaiaGoFn)(const char *, uintptr_t, uintptr_t, MaiaCallback);
static MaiaGoFn originalMaiaGo = nullptr;
static NSString *lastScheduledFEN = nil;
static NSString *lastPlayedFEN = nil;
static uint64_t autoPlayGeneration = 0; // Main queue only.

typedef NS_ENUM(NSInteger, CAPMoveSource) {
    CAPMoveSourceMaia,
    CAPMoveSourceGod
};

typedef NS_ENUM(NSInteger, CAPAutoEngine) {
    CAPAutoEngineMaia,
    CAPAutoEngineStockfish19,
    CAPAutoEngineTriumviratus
};

static NSString *lastObservedPosition = nil; // Main queue only.
static NSString *cachedMaiaMove = nil, *cachedMaiaFEN = nil;
static NSString *cachedGodMove = nil, *cachedGodFEN = nil;
static NSInteger cachedGodChoice = -1;
static NSString *CAPPositionKey(NSString *fen);

static CAPAutoEngine CAPSelectedAutoEngine(void) {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    if (![defaults objectForKey:kAutoEngineKey])
        return [defaults boolForKey:@"ChessAssist_UseMaia"] ? CAPAutoEngineMaia :
            ([defaults integerForKey:kGodEngineKey] == 1 ?
             CAPAutoEngineTriumviratus : CAPAutoEngineStockfish19);
    NSInteger value = [defaults integerForKey:kAutoEngineKey];
    return value >= CAPAutoEngineMaia && value <= CAPAutoEngineTriumviratus ?
        (CAPAutoEngine)value : CAPAutoEngineMaia;
}

static void CAPObservePosition(NSString *fen) {
    NSString *key = CAPPositionKey(fen);
    if (!key) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (![lastObservedPosition isEqualToString:key]) {
            lastObservedPosition = [key copy];
            ++autoPlayGeneration;
            lastScheduledFEN = nil;
            cachedMaiaMove = cachedMaiaFEN = nil;
            cachedGodMove = cachedGodFEN = nil;
            cachedGodChoice = -1;
        }
    });
}

static NSString *CAPNormalizeMove(NSString *move) {
    if (!move) return nil;
    NSMutableString *normalized = [NSMutableString stringWithCapacity:5];
    NSCharacterSet *allowed = NSCharacterSet.alphanumericCharacterSet;
    for (NSUInteger i = 0; i < move.length; ++i) {
        unichar c = [move characterAtIndex:i];
        if ([allowed characterIsMember:c]) [normalized appendFormat:@"%C", c];
    }
    return normalized.lowercaseString;
}

// Compare the position, turn, castling rights, and en-passant square. Ignore
// only the halfmove and fullmove counters, which do not change legal moves.
static NSString *CAPPositionKey(NSString *fen) {
    if (!fen.length) return nil;
    NSArray<NSString *> *parts = [fen componentsSeparatedByCharactersInSet:
        [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSMutableArray<NSString *> *fields = [NSMutableArray arrayWithCapacity:4];
    for (NSString *part in parts) {
        if (part.length && fields.count < 4) [fields addObject:part];
    }
    return fields.count == 4 ? [fields componentsJoinedByString:@" "] : nil;
}

static BOOL CAPSourceEnabled(CAPMoveSource source) {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    id assistantEnabled = [defaults objectForKey:@"ChessAssist_Enabled"];
    if (assistantEnabled && ![assistantEnabled boolValue]) return NO;
    CAPAutoEngine selected = CAPSelectedAutoEngine();
    BOOL maiaSelected = [defaults boolForKey:@"ChessAssist_UseMaia"];
    return source == CAPMoveSourceMaia ?
        (selected == CAPAutoEngineMaia && maiaSelected) :
        (selected != CAPAutoEngineMaia);
}

static double CAPDelay(NSString *key, double fallback) {
    id stored = [[NSUserDefaults standardUserDefaults] objectForKey:key];
    double value = stored ? [stored doubleValue] : fallback;
    return isfinite(value) ? fmin(3.0, fmax(0.000001, value)) : fallback;
}

static BOOL CAPBoardMatchesPosition(id board, id game, NSString *positionKey) {
    SEL selector = @selector(fenForCurrentPosition:flipped:);
    for (id source in @[game, board]) {
        if (![source respondsToSelector:selector]) continue;
        id value = ((id (*)(id, SEL, BOOL, BOOL))objc_msgSend)(source, selector, YES, NO);
        if ([value isKindOfClass:NSString.class] &&
            [CAPPositionKey(value) isEqualToString:positionKey]) return YES;
    }
    return NO;
}

static UIView *CAPFindMatchingBoard(UIView *view, NSString *positionKey, BOOL whiteToMove) {
    if (view.hidden || view.alpha < 0.05) return nil;
    Class boardClass = NSClassFromString(@"CHBoardView");
    Class analysisClass = NSClassFromString(@"CHAnalysisBoardView");
    if (boardClass && [view isKindOfClass:boardClass] &&
        !(analysisClass && [view isKindOfClass:analysisClass]) &&
        view.window && CGRectIntersectsRect([view convertRect:view.bounds toView:view.window],
                                            view.window.bounds) &&
        [view respondsToSelector:@selector(chessGame)] &&
        [view respondsToSelector:@selector(makeMove:throughTouches:)]) {
        SEL permission = whiteToMove ? @selector(allowUserToMakeWhiteMoves)
                                     : @selector(allowUserToMakeBlackMoves);
        id game = ((id (*)(id, SEL))objc_msgSend)(view, @selector(chessGame));
        if (game && [view respondsToSelector:permission] &&
            ((BOOL (*)(id, SEL))objc_msgSend)(view, permission) &&
            CAPBoardMatchesPosition(view, game, positionKey))
            return view;
    }
    for (UIView *child in view.subviews) {
        UIView *found = CAPFindMatchingBoard(child, positionKey, whiteToMove);
        if (found) return found;
    }
    return nil;
}

static UIView *CAPVisibleBoardForFEN(NSString *fen) {
    NSString *key = CAPPositionKey(fen);
    if (!key) return nil;
    BOOL whiteToMove = [fen containsString:@" w "];
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class] ||
            scene.activationState != UISceneActivationStateForegroundActive) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            UIView *board = CAPFindMatchingBoard(window, key, whiteToMove);
            if (board) return board;
        }
    }
    return nil;
}

static BOOL CAPMoveWord(NSString *name) {
    return [name rangeOfString:@"move" options:NSCaseInsensitiveSearch].location != NSNotFound &&
           [name rangeOfString:@"remove" options:NSCaseInsensitiveSearch].location == NSNotFound;
}

static void CAPDumpClass(Class cls) {
    if (!cls) return;
    CAPDiscLog(@"CLASS %@ image=%s", NSStringFromClass(cls), class_getImageName(cls) ?: "?");
    unsigned count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned i = 0; i < count; ++i)
        CAPDiscLog(@"  METHOD %s encoding=%s imp=%p",
                   sel_getName(method_getName(methods[i])),
                   method_getTypeEncoding(methods[i]) ?: "?",
                   method_getImplementation(methods[i]));
    free(methods);
    Ivar *ivars = class_copyIvarList(cls, &count);
    for (unsigned i = 0; i < count; ++i)
        CAPDiscLog(@"  IVAR %s encoding=%s offset=%td",
                   ivar_getName(ivars[i]), ivar_getTypeEncoding(ivars[i]) ?: "?",
                   ivar_getOffset(ivars[i]));
    free(ivars);
    objc_property_t *properties = class_copyPropertyList(cls, &count);
    for (unsigned i = 0; i < count; ++i)
        CAPDiscLog(@"  PROPERTY %s attributes=%s", property_getName(properties[i]),
                   property_getAttributes(properties[i]) ?: "?");
    free(properties);
}

static void CAPDumpSwiftSymbols(const struct mach_header_64 *header) {
    if (!header || header->magic != MH_MAGIC_64) {
        CAPDiscLog(@"SYMBOLS no Mach-O header");
        return;
    }
    const struct symtab_command *symtab = nullptr;
    const struct segment_command_64 *linkedit = nullptr, *textSegment = nullptr;
    const uint8_t *cursor = (const uint8_t *)header + sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command = (const struct load_command *)cursor;
        if (command->cmdsize < sizeof(*command)) break;
        if (command->cmd == LC_SYMTAB) symtab = (const struct symtab_command *)cursor;
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment = (const struct segment_command_64 *)cursor;
            if (strcmp(segment->segname, "__LINKEDIT") == 0) linkedit = segment;
            if (strcmp(segment->segname, "__TEXT") == 0) textSegment = segment;
        }
        cursor += command->cmdsize;
    }
    if (!symtab || !linkedit || !textSegment ||
        symtab->symoff < linkedit->fileoff || symtab->stroff < linkedit->fileoff ||
        (uint64_t)symtab->symoff + (uint64_t)symtab->nsyms * sizeof(struct nlist_64) >
            linkedit->fileoff + linkedit->filesize ||
        (uint64_t)symtab->stroff + symtab->strsize > linkedit->fileoff + linkedit->filesize) {
        CAPDiscLog(@"SYMBOLS LC_SYMTAB unavailable or outside __LINKEDIT");
        return;
    }
    intptr_t slide = (intptr_t)header - (intptr_t)textSegment->vmaddr;
    uintptr_t base = (uintptr_t)(slide + (intptr_t)linkedit->vmaddr - (intptr_t)linkedit->fileoff);
    const struct nlist_64 *symbols = (const struct nlist_64 *)(base + symtab->symoff);
    const char *strings = (const char *)(base + symtab->stroff);
    unsigned matches = 0;
    for (uint32_t i = 0; i < symtab->nsyms; ++i) {
        uint32_t index = symbols[i].n_un.n_strx;
        if (index >= symtab->strsize) continue;
        const char *name = strings + index;
        size_t length = strnlen(name, symtab->strsize - index);
        if (length >= symtab->strsize - index) continue;
        std::string symbol(name, length), lower = symbol;
        for (char &c : lower) c = (char)tolower((unsigned char)c);
        if (symbol.find("SwiftChessBoard") == std::string::npos) continue;
        if (lower.find("move") == std::string::npos && lower.find("drop") == std::string::npos &&
            lower.find("select") == std::string::npos && lower.find("tap") == std::string::npos &&
            lower.find("piece") == std::string::npos && lower.find("delegate") == std::string::npos &&
            lower.find("completion") == std::string::npos) continue;
        uintptr_t address = symbols[i].n_value ? (uintptr_t)(slide + symbols[i].n_value) : 0;
        CAPDiscLog(@"SYMBOL %s address=%p type=0x%x", name, (void *)address, symbols[i].n_type);
        ++matches;
    }
    CAPDiscLog(@"SYMBOLS scanned=%u matched=%u", symtab->nsyms, matches);
}

static NSString *CAPMoveArg(id value) {
    if (!value) return @"nil";
    if ([value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class])
        return [NSString stringWithFormat:@"%@", value];
    NSString *move = nil;
    NSMethodSignature *signature = [value respondsToSelector:@selector(lanMove)] ?
        [value methodSignatureForSelector:@selector(lanMove)] : nil;
    if (signature && signature.methodReturnType[0] == '@') {
        @try { move = ((id (*)(id, SEL))objc_msgSend)(value, @selector(lanMove)); }
        @catch (NSException *exception) {}
    }
    return [NSString stringWithFormat:@"<%@ %p%@>", NSStringFromClass([value class]), value,
            [move isKindOfClass:NSString.class] ? [@" lan=" stringByAppendingString:move] : @""];
}

static NSMutableDictionary<NSString *, NSValue *> *CAPMoveOriginals(void) {
    static NSMutableDictionary<NSString *, NSValue *> *originals;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ originals = [NSMutableDictionary dictionary]; });
    return originals;
}

static IMP CAPMoveOriginal(SEL selector) {
    @synchronized (CAPMoveOriginals()) {
        return (IMP)[CAPMoveOriginals()[NSStringFromSelector(selector)] pointerValue];
    }
}

static void CAPObserveMove0(id self, SEL selector) {
    CAPDiscLog(@"CALL %@ %@", NSStringFromClass([self class]), NSStringFromSelector(selector));
    void (*original)(id, SEL) = (void (*)(id, SEL))CAPMoveOriginal(selector);
    if (original) original(self, selector);
}
static void CAPObserveMove1(id self, SEL selector, id arg) {
    CAPDiscLog(@"CALL %@ %@ arg=%@", NSStringFromClass([self class]),
               NSStringFromSelector(selector), CAPMoveArg(arg));
    void (*original)(id, SEL, id) = (void (*)(id, SEL, id))CAPMoveOriginal(selector);
    if (original) original(self, selector, arg);
}
static void CAPObserveMove1Bool(id self, SEL selector, id arg, BOOL flag) {
    CAPDiscLog(@"CALL %@ %@ arg=%@ flag=%d", NSStringFromClass([self class]),
               NSStringFromSelector(selector), CAPMoveArg(arg), flag);
    void (*original)(id, SEL, id, BOOL) =
        (void (*)(id, SEL, id, BOOL))CAPMoveOriginal(selector);
    if (original) original(self, selector, arg, flag);
}
static void CAPObserveMove2(id self, SEL selector, id first, id second) {
    CAPDiscLog(@"CALL %@ %@ first=%@ second=%@", NSStringFromClass([self class]),
               NSStringFromSelector(selector), CAPMoveArg(first), CAPMoveArg(second));
    void (*original)(id, SEL, id, id) =
        (void (*)(id, SEL, id, id))CAPMoveOriginal(selector);
    if (original) original(self, selector, first, second);
}

static void CAPInstallMoveObservers(Class cls) {
    if (!cls) return;
    unsigned count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned i = 0; i < count; ++i) {
        Method method = methods[i];
        SEL selector = method_getName(method);
        NSString *name = NSStringFromSelector(selector);
        if (!CAPMoveWord(name)) continue;
        const char *types = method_getTypeEncoding(method);
        NSMethodSignature *signature = types ? [NSMethodSignature signatureWithObjCTypes:types] : nil;
        if (!signature || signature.methodReturnType[0] != 'v') continue;
        NSUInteger arguments = signature.numberOfArguments;
        IMP replacement = nullptr;
        if (arguments == 2) replacement = (IMP)CAPObserveMove0;
        else if (arguments == 3 && [signature getArgumentTypeAtIndex:2][0] == '@')
            replacement = (IMP)CAPObserveMove1;
        else if (arguments == 4 && [signature getArgumentTypeAtIndex:2][0] == '@') {
            char last = [signature getArgumentTypeAtIndex:3][0];
            if (last == '@') replacement = (IMP)CAPObserveMove2;
            else if (last == 'B' || last == 'c') replacement = (IMP)CAPObserveMove1Bool;
        }
        if (!replacement) {
            CAPDiscLog(@"OBSERVER skipped %@ %@ encoding=%s (ABI not verified)",
                       NSStringFromClass(cls), name, types ?: "?");
            continue;
        }
        @synchronized (CAPMoveOriginals()) {
            if (CAPMoveOriginals()[name]) {
                CAPDiscLog(@"OBSERVER skipped duplicate selector %@ on %@", name, NSStringFromClass(cls));
                continue;
            }
            IMP previous = method_getImplementation(method);
            CAPMoveOriginals()[name] = [NSValue valueWithPointer:(const void *)previous];
            IMP original = nullptr;
            MSHookMessageEx(cls, selector, replacement, &original);
            if (original) CAPMoveOriginals()[name] =
                [NSValue valueWithPointer:(const void *)original];
            CAPDiscLog(@"OBSERVER registered %@ %@ encoding=%s original=%p",
                       NSStringFromClass(cls), name, types, original ?: previous);
        }
    }
    free(methods);
}

static void CAPDumpGesture(UIGestureRecognizer *gesture) {
    CAPDiscLog(@"    GESTURE %@ %p state=%ld enabled=%d",
               NSStringFromClass([gesture class]), gesture,
               (long)gesture.state, gesture.enabled);
    @try {
        id rawTargets = [gesture valueForKey:@"_targets"];
        id<NSFastEnumeration> targets =
            ([rawTargets isKindOfClass:NSArray.class] || [rawTargets isKindOfClass:NSSet.class]) ?
                rawTargets : nil;
        if (!targets) {
            CAPDiscLog(@"      _targets unavailable: %@", CAPMoveArg(rawTargets));
            return;
        }
        for (id entry in targets) {
            Ivar targetIvar = class_getInstanceVariable([entry class], "_target");
            Ivar actionIvar = class_getInstanceVariable([entry class], "_action");
            const char *targetType = targetIvar ? ivar_getTypeEncoding(targetIvar) : nullptr;
            const char *actionType = actionIvar ? ivar_getTypeEncoding(actionIvar) : nullptr;
            id target = targetType && targetType[0] == '@' ?
                object_getIvar(entry, targetIvar) : nil;
            SEL action = nullptr;
            if (actionType && actionType[0] == ':')
                memcpy(&action, (const uint8_t *)(__bridge const void *)entry +
                       ivar_getOffset(actionIvar), sizeof(action));
            CAPDiscLog(@"      TARGET %@ %p action=%s", NSStringFromClass([target class]),
                       target, action ? sel_getName(action) : "?");
        }
    } @catch (NSException *exception) {
        CAPDiscLog(@"      _targets KVC failed: %@", exception.name);
    }
}

static void CAPDumpSubviewTree(UIView *view, NSUInteger depth, NSUInteger *count) {
    if (!view || depth > 8 || *count >= 2000) return;
    ++*count;
    CAPDiscLog(@"VIEW depth=%lu %@ %p frame=%@ hidden=%d alpha=%.2f",
               (unsigned long)depth, NSStringFromClass([view class]), view,
               NSStringFromCGRect(view.frame), view.hidden, view.alpha);
    for (UIGestureRecognizer *gesture in view.gestureRecognizers)
        CAPDumpGesture(gesture);
    for (UIView *child in view.subviews)
        CAPDumpSubviewTree(child, depth + 1, count);
}

static void CAPDumpResponderChain(UIView *board) {
    UIResponder *responder = board;
    for (NSUInteger depth = 0; responder && depth < 24; ++depth) {
        Class cls = [responder class];
        CAPDiscLog(@"RESPONDER depth=%lu %@ %p", (unsigned long)depth,
                   NSStringFromClass(cls), responder);
        unsigned count = 0;
        Method *methods = class_copyMethodList(cls, &count);
        for (unsigned i = 0; i < count; ++i) {
            NSString *name = NSStringFromSelector(method_getName(methods[i]));
            if (CAPMoveWord(name) ||
                [name rangeOfString:@"game" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                [name rangeOfString:@"board" options:NSCaseInsensitiveSearch].location != NSNotFound)
                CAPDiscLog(@"  RESPONDER METHOD %@ encoding=%s", name,
                           method_getTypeEncoding(methods[i]) ?: "?");
        }
        free(methods);
        const char *image = class_getImageName(cls);
        if (image && strstr(image, "/Chess.app/")) CAPInstallMoveObservers(cls);
        if ([responder isKindOfClass:UIViewController.class]) break;
        responder = responder.nextResponder;
    }
}

static void CAPDumpObjectIvars(UIView *board) {
    unsigned count = 0;
    Ivar *ivars = class_copyIvarList([board class], &count);
    for (unsigned i = 0; i < count; ++i) {
        const char *encoding = ivar_getTypeEncoding(ivars[i]);
        if (!encoding || encoding[0] != '@') continue;
        @try {
            id value = object_getIvar(board, ivars[i]);
            CAPDiscLog(@"OBJECT IVAR %s -> %@ %p", ivar_getName(ivars[i]),
                       NSStringFromClass([value class]), value);
        } @catch (NSException *exception) {
            CAPDiscLog(@"OBJECT IVAR %s read failed: %@", ivar_getName(ivars[i]),
                       exception.name);
        }
    }
    free(ivars);
}

static NSString *CAPBoardGameKey(UIView *board) {
    for (UIResponder *item = board; item; item = item.nextResponder) {
        SEL gameID = @selector(gameID);
        if ([item respondsToSelector:gameID]) {
            NSMethodSignature *signature = [item methodSignatureForSelector:gameID];
            const char *type = signature ? signature.methodReturnType : nullptr;
            @try {
                if (type && (type[0] == 'Q' || type[0] == 'q')) {
                    long long identifier = ((long long (*)(id, SEL))objc_msgSend)(item, gameID);
                    if (identifier) return [NSString stringWithFormat:@"game:%lld", identifier];
                } else if (type && type[0] == '@') {
                    id identifier = ((id (*)(id, SEL))objc_msgSend)(item, gameID);
                    if ([identifier isKindOfClass:NSString.class] ||
                        [identifier isKindOfClass:NSNumber.class])
                        return [NSString stringWithFormat:@"game:%@", identifier];
                }
            } @catch (NSException *exception) {}
        }
        if ([item respondsToSelector:@selector(game)]) {
            NSMethodSignature *signature = [item methodSignatureForSelector:@selector(game)];
            if (signature && signature.methodReturnType[0] == '@') {
                @try {
                    id game = ((id (*)(id, SEL))objc_msgSend)(item, @selector(game));
                    if (game) return [NSString stringWithFormat:@"game-object:%p", game];
                } @catch (NSException *exception) {}
            }
        }
    }
    return [NSString stringWithFormat:@"board:%p", board];
}

static void CAPDumpSwiftBoard(UIView *board, BOOL forced) {
    if (!board || ![NSThread isMainThread]) return;
    Class cls = [board class];
    Dl_info info = {};
    BOOL located = dladdr((__bridge const void *)cls, &info) != 0;
    const char *image = located && info.dli_fname ? info.dli_fname : class_getImageName(cls);
    const struct mach_header_64 *header =
        located ? (const struct mach_header_64 *)info.dli_fbase : nullptr;
    if (!header && image) {
        for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
            const char *candidate = _dyld_get_image_name(i);
            if (candidate && strcmp(candidate, image) == 0) {
                header = (const struct mach_header_64 *)_dyld_get_image_header(i);
                break;
            }
        }
    }
    CAPDiscLog(@"===== SWIFT BOARD DUMP %@ key=%@ board=%p image=%s file=%@ =====",
               forced ? @"forced" : @"automatic", CAPBoardGameKey(board), board,
               image ?: "?", CAPDumpPath());
    for (Class current = cls; current; current = class_getSuperclass(current)) {
        CAPDumpClass(current);
        if (current == NSObject.class) break;
    }
    if (image) {
        unsigned classCount = 0;
        const char **names = objc_copyClassNamesForImage(image, &classCount);
        CAPDiscLog(@"MODULE image=%s classes=%u", image, classCount);
        for (unsigned i = 0; i < classCount; ++i) {
            if (strncmp(names[i], "_TtC15SwiftChessBoard", 21) != 0) continue;
            Class member = objc_getClass(names[i]);
            CAPDumpClass(member);
            CAPInstallMoveObservers(member);
        }
        free(names);
    }
    CAPDumpSwiftSymbols(header);
    NSUInteger viewCount = 0;
    CAPDumpSubviewTree(board, 0, &viewCount);
    CAPDiscLog(@"VIEWS logged=%lu", (unsigned long)viewCount);
    CAPDumpResponderChain(board);
    CAPDumpObjectIvars(board);
    @synchronized ([NSURLSessionWebSocketTask class]) {
        CAPDiscLog(@"LAST OUTGOING MOVE %@", capLastOutgoingMove ?: @"none");
    }
    CAPDiscLog(@"===== END SWIFT BOARD DUMP =====");
}


// Share the discovery dump (the sideloaded app has no Files-app access).
static UIWindow *CAPDiscKeyWindow(void) {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) continue;
        for (UIWindow *w in ((UIWindowScene *)scene).windows) if (w.isKeyWindow) return w;
    }
    return nil;
}

@interface CAPShareHelper : NSObject
+ (void)tapped;
@end
@implementation CAPShareHelper
+ (void)tapped {
    NSString *path = CAPDumpPath();
    if (!path || ![[NSFileManager defaultManager] fileExistsAtPath:path]) return;
    UIViewController *top = CAPDiscKeyWindow().rootViewController;
    while (top.presentedViewController) top = top.presentedViewController;
    if (!top) return;
    UIActivityViewController *vc = [[UIActivityViewController alloc]
        initWithActivityItems:@[[NSURL fileURLWithPath:path]] applicationActivities:nil];
    vc.popoverPresentationController.sourceView = top.view;
    vc.popoverPresentationController.sourceRect =
        CGRectMake(CGRectGetMidX(top.view.bounds), 80, 1, 1);
    [top presentViewController:vc animated:YES completion:nil];
}
@end

static void CAPAddShareButton(UIWindow *window) {
    static BOOL added = NO;
    if (added || !window) return;
    added = YES;
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    [button setTitle:@"Share dump" forState:UIControlStateNormal];
    [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    button.backgroundColor = [UIColor colorWithWhite:0 alpha:0.7];
    button.layer.cornerRadius = 8;
    button.frame = CGRectMake(8, 60, 110, 32);
    [button addTarget:CAPShareHelper.class action:@selector(tapped)
     forControlEvents:UIControlEventTouchUpInside];
    [window addSubview:button];
}

static void CAPMaybeDumpSwiftBoard(UIView *board) {
    if (!board || ![NSThread isMainThread]) return;
    static NSMapTable<UIView *, NSString *> *keys;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ keys = [NSMapTable weakToStrongObjectsMapTable]; });
    NSString *key = CAPBoardGameKey(board);
    if ([[keys objectForKey:board] isEqualToString:key]) return;
    [keys setObject:key forKey:board];
    @try { CAPDumpSwiftBoard(board, NO); }
    @catch (NSException *exception) {
        CAPDiscLog(@"DUMP failed: %@ %@", exception.name, exception.reason);
    }
    CAPAddShareButton(board.window);
}

typedef void (*CAPWebSocketSendIMP)(NSURLSessionWebSocketTask *, SEL,
                                    NSURLSessionWebSocketMessage *,
                                    void (^)(NSError *));
static CAPWebSocketSendIMP originalWebSocketSend = nullptr;
static void CAPWebSocketSend(NSURLSessionWebSocketTask *task, SEL selector,
                             NSURLSessionWebSocketMessage *message,
                             void (^completion)(NSError *)) {
    if (message.type == NSURLSessionWebSocketMessageTypeString) {
        NSString *payload = message.string;
        static NSRegularExpression *movePattern;
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            movePattern = [NSRegularExpression regularExpressionWithPattern:
                @"\\b(?:[a-h][1-8][a-h][1-8][qrbn]?|move)\\b"
                options:NSRegularExpressionCaseInsensitive error:nil];
        });
        if (payload && [movePattern firstMatchInString:payload options:0
                                                range:NSMakeRange(0, payload.length)]) {
            @synchronized ([NSURLSessionWebSocketTask class]) {
                capLastOutgoingMove = [payload copy];
            }
            CAPDiscLog(@"WEBSOCKET OUT task=%p message=%@", task, payload);
        }
    }
    originalWebSocketSend(task, selector, message, completion);
}

// Detect the native Swift board so an unsupported move path fails explicitly.
static UIView *CAPFindSwiftBoard(UIView *view) {
    if (view.hidden || view.alpha < 0.05 || !view.window) return nil;
    Class swiftBoard = NSClassFromString(@"_TtC15SwiftChessBoard10ChessBoard");
    if (swiftBoard && [view isKindOfClass:swiftBoard] &&
        view.bounds.size.width > 160 &&
        fabs(view.bounds.size.width - view.bounds.size.height) < 4 &&
        CGRectIntersectsRect([view convertRect:view.bounds toView:view.window], view.window.bounds))
        return view;
    for (UIView *child in view.subviews) {
        UIView *found = CAPFindSwiftBoard(child);
        if (found) return found;
    }
    return nil;
}

static UIView *CAPVisibleSwiftBoard(void) {
    UIView *selected = nil;
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class] ||
            scene.activationState != UISceneActivationStateForegroundActive) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            UIView *found = CAPFindSwiftBoard(window);
            if (!found) continue;
            if (selected && selected != found) {
                NSLog(@"[ChessAssistantPro] Auto-Play [board] multiple visible Swift boards");
                return nil;
            }
            selected = found;
        }
    }
    CAPMaybeDumpSwiftBoard(selected);
    return selected;
}


// ---- Read-only observers on the Swift board's real touch handlers ----
typedef void (*CAPTouchIMP)(UIView *, SEL, NSSet<UITouch *> *, UIEvent *);
static CAPTouchIMP capOrigTouchBegan = nullptr, capOrigTouchMoved = nullptr,
                   capOrigTouchEnded = nullptr;
static NSUInteger capTouchMovedCount = 0;

static void CAPLogTouches(NSString *phase, UIView *view, NSSet<UITouch *> *touches,
                          UIEvent *event) {
    for (UITouch *touch in touches) {
        CGPoint p = [touch locationInView:view], q = [touch previousLocationInView:view];
        CAPDiscLog(@"TOUCH %@ view=%@ %p touchClass=%@ loc=%@ prev=%@ ts=%.4f tapCount=%lu "
                   @"radius=%.2f type=%ld phase=%ld touchView=%@ gestures=%lu eventClass=%@ "
                   @"eventType=%ld allTouches=%lu",
                   phase, NSStringFromClass([view class]), view,
                   NSStringFromClass([touch class]), NSStringFromCGPoint(p),
                   NSStringFromCGPoint(q), touch.timestamp, (unsigned long)touch.tapCount,
                   touch.majorRadius, (long)touch.type, (long)touch.phase,
                   NSStringFromClass([touch.view class]),
                   (unsigned long)touch.gestureRecognizers.count,
                   NSStringFromClass([event class]), (long)event.type,
                   (unsigned long)event.allTouches.count);
    }
}

static void CAPLogPieceViewsIn(UIView *view, UIView *overlay, CGFloat square,
                               Class pieceClass, NSUInteger *count) {
    if (!view || *count >= 80) return;
    if ([view isKindOfClass:pieceClass] && view.superview) {
        CGPoint c = [view.superview convertPoint:view.center toView:overlay];
        if (CGRectContainsPoint(overlay.bounds, c) && square > 0) {
            CAPDiscLog(@"PIECE %p col=%d rowFromTop=%d center=%@ hidden=%d alpha=%.2f",
                       view, (int)floor(c.x / square), (int)floor(c.y / square),
                       NSStringFromCGPoint(c), view.hidden, view.alpha);
            ++*count;
        }
    }
    for (UIView *child in view.subviews)
        CAPLogPieceViewsIn(child, overlay, square, pieceClass, count);
}

static void CAPLogPieceViews(UIView *overlay) {
    Class pieceClass = NSClassFromString(@"_TtC15SwiftChessBoard9PieceView");
    if (!overlay || !overlay.window || !pieceClass) return;
    NSUInteger count = 0;
    CAPDiscLog(@"PIECES after touch (overlay=%@)", NSStringFromCGRect(overlay.bounds));
    CAPLogPieceViewsIn(overlay.window, overlay, overlay.bounds.size.width / 8.0,
                       pieceClass, &count);
    CAPDiscLog(@"PIECES logged=%lu", (unsigned long)count);
}

static void CAPOverlayTouchBegan(UIView *self, SEL selector, NSSet<UITouch *> *touches,
                                 UIEvent *event) {
    capTouchMovedCount = 0;
    @try { CAPLogTouches(@"began", self, touches, event); } @catch (NSException *e) {}
    if (capOrigTouchBegan) capOrigTouchBegan(self, selector, touches, event);
}
static void CAPOverlayTouchMoved(UIView *self, SEL selector, NSSet<UITouch *> *touches,
                                 UIEvent *event) {
    if (++capTouchMovedCount % 5 == 1) {
        @try { CAPLogTouches(@"moved", self, touches, event); } @catch (NSException *e) {}
    }
    if (capOrigTouchMoved) capOrigTouchMoved(self, selector, touches, event);
}
static void CAPOverlayTouchEnded(UIView *self, SEL selector, NSSet<UITouch *> *touches,
                                 UIEvent *event) {
    @try { CAPLogTouches(@"ended", self, touches, event); } @catch (NSException *e) {}
    if (capOrigTouchEnded) capOrigTouchEnded(self, selector, touches, event);
    __weak UIView *weakOverlay = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 400 * NSEC_PER_MSEC),
                   dispatch_get_main_queue(), ^{
        UIView *overlay = weakOverlay;
        @try { CAPLogPieceViews(overlay); } @catch (NSException *e) {}
    });
}

static void CAPInstallTouchObservers(void) {
    static BOOL installed = NO;
    if (installed) return;
    Class overlay = NSClassFromString(@"_TtC15SwiftChessBoard28PiecesOverlayWithTouchEvents");
    if (!overlay) return;
    installed = YES;
    MSHookMessageEx(overlay, @selector(touchesBegan:withEvent:),
                    (IMP)CAPOverlayTouchBegan, (IMP *)&capOrigTouchBegan);
    MSHookMessageEx(overlay, @selector(touchesMoved:withEvent:),
                    (IMP)CAPOverlayTouchMoved, (IMP *)&capOrigTouchMoved);
    MSHookMessageEx(overlay, @selector(touchesEnded:withEvent:),
                    (IMP)CAPOverlayTouchEnded, (IMP *)&capOrigTouchEnded);
    CAPDiscLog(@"touch observers installed on %@", NSStringFromClass(overlay));
}

typedef void (*CAPLayoutIMP)(UIView *, SEL);
static CAPLayoutIMP originalSwiftBoardLayout = nullptr;
static BOOL capBoardProbeQueued = NO; // UIKit layout runs on the main thread.
static void CAPSwiftBoardLayout(UIView *board, SEL selector) {
    originalSwiftBoardLayout(board, selector);
    CAPInstallTouchObservers();
    if (!board.window || capBoardProbeQueued) return;
    capBoardProbeQueued = YES;
    dispatch_async(dispatch_get_main_queue(), ^{
        capBoardProbeQueued = NO;
        CAPVisibleSwiftBoard();
    });
}

// ---- Bot-only Auto-Play for the Swift board (synthetic touches on the pieces overlay) ----
@interface CAPFakeTouch : UITouch
@property (nonatomic, weak) UIView *capView;
@property (nonatomic, assign) CGPoint capPoint, capPrev;
@property (nonatomic, assign) UITouchPhase capPhase;
@property (nonatomic, assign) NSTimeInterval capTime;
@end
static CGPoint CAPConvertPoint(UIView *src, CGPoint p, UIView *dst) {
    if (!src || dst == src) return p;
    return [src convertPoint:p toView:dst];
}
@implementation CAPFakeTouch
- (CGPoint)locationInView:(UIView *)v { return CAPConvertPoint(self.capView, self.capPoint, v); }
- (CGPoint)previousLocationInView:(UIView *)v { return CAPConvertPoint(self.capView, self.capPrev, v); }
- (CGPoint)preciseLocationInView:(UIView *)v { return [self locationInView:v]; }
- (CGPoint)precisePreviousLocationInView:(UIView *)v { return [self previousLocationInView:v]; }
- (UITouchPhase)phase { return self.capPhase; }
- (NSUInteger)tapCount { return 1; }
- (UIView *)view { return self.capView; }
- (UIWindow *)window { return self.capView.window; }
- (NSTimeInterval)timestamp { return self.capTime; }
- (UITouchType)type { return UITouchTypeDirect; }
- (CGFloat)majorRadius { return 20; }
- (CGFloat)force { return 0; }
- (CGFloat)maximumPossibleForce { return 1; }
- (NSArray *)gestureRecognizers { return @[]; }
@end

@interface CAPFakeEvent : UIEvent
@property (nonatomic, strong) NSSet *capTouches;
@property (nonatomic, assign) NSTimeInterval capTime;
@end
@implementation CAPFakeEvent
- (NSSet *)allTouches { return self.capTouches; }
- (NSSet *)touchesForView:(UIView *)v { return self.capTouches; }
- (NSSet *)touchesForWindow:(UIWindow *)w { return self.capTouches; }
- (UIEventType)type { return UIEventTypeTouches; }
- (NSTimeInterval)timestamp { return self.capTime; }
@end

static UIView *CAPFindOverlay(UIView *view, Class overlayClass) {
    if (view.hidden || view.alpha < 0.05 || !view.window) return nil;
    if ([view isKindOfClass:overlayClass] && view.bounds.size.width > 160 &&
        CGRectIntersectsRect([view convertRect:view.bounds toView:view.window], view.window.bounds))
        return view;
    for (UIView *child in view.subviews) {
        UIView *found = CAPFindOverlay(child, overlayClass);
        if (found) return found;
    }
    return nil;
}

static void CAPCollectPieces(UIView *view, Class pieceClass, NSMutableArray *out) {
    for (UIView *child in view.subviews) {
        if ([child isKindOfClass:pieceClass] && !child.hidden && child.alpha > 0.05) [out addObject:child];
        CAPCollectPieces(child, pieceClass, out);
    }
}

static BOOL CAPClassChainHasBot(UIViewController *vc, int depth) {
    if (!vc || depth > 12) return NO;
    if ([NSStringFromClass(vc.class) containsString:@"PlayBot"]) return YES;
    if ([vc isKindOfClass:UINavigationController.class]) {
        if (CAPClassChainHasBot(((UINavigationController *)vc).topViewController, depth + 1)) return YES;
    }
    if ([vc isKindOfClass:UITabBarController.class]) {
        if (CAPClassChainHasBot(((UITabBarController *)vc).selectedViewController, depth + 1)) return YES;
    }
    if (CAPClassChainHasBot(vc.presentedViewController, depth + 1)) return YES;
    for (UIViewController *child in vc.childViewControllers)
        if (CAPClassChainHasBot(child, depth + 1)) return YES;
    return NO;
}

// YES only when the visible game screen belongs to the PlayBot module (computer opponent).
static BOOL CAPIsBotGameScreen(UIView *overlay) {
    for (UIResponder *r = overlay; r; r = r.nextResponder)
        if ([NSStringFromClass(r.class) containsString:@"PlayBot"]) return YES;
    return CAPClassChainHasBot(overlay.window.rootViewController, 0);
}

static CGPoint CAPSquareCenter(CGFloat size, int file, int rank, BOOL flipped) {
    CGFloat sq = size / 8.0;
    CGFloat x = (flipped ? 7 - file : file) + 0.5;
    CGFloat y = (flipped ? rank : 7 - rank) + 0.5;
    return CGPointMake(x * sq, y * sq);
}

static uint64_t CAPFenMask(NSString *fen) {
    uint64_t mask = 0; int rank = 7, file = 0;
    NSString *board = [fen componentsSeparatedByString:@" "].firstObject;
    for (NSUInteger i = 0; i < board.length; i++) {
        unichar c = [board characterAtIndex:i];
        if (c == '/') { rank--; file = 0; }
        else if (c >= '1' && c <= '8') file += c - '0';
        else { if (rank >= 0 && file < 8) mask |= (1ULL << (rank * 8 + file)); file++; }
    }
    return mask;
}

static uint64_t CAPPieceMask(NSArray *pieces, UIView *overlay, BOOL flipped) {
    uint64_t mask = 0; CGFloat sq = overlay.bounds.size.width / 8.0;
    for (UIView *p in pieces) {
        if (!p.superview) continue;
        CGPoint c = [p.superview convertPoint:p.center toView:overlay];
        int f = (int)floor(c.x / sq), rt = (int)floor(c.y / sq);
        if (f < 0 || f > 7 || rt < 0 || rt > 7) continue;
        int file = flipped ? 7 - f : f, rank = flipped ? rt : 7 - rt;
        mask |= (1ULL << (rank * 8 + file));
    }
    return mask;
}

static BOOL CAPPieceAt(NSArray *pieces, UIView *overlay, int file, int rank, BOOL flipped) {
    CGPoint target = CAPSquareCenter(overlay.bounds.size.width, file, rank, flipped);
    CGFloat sq = overlay.bounds.size.width / 8.0;
    for (UIView *p in pieces) {
        if (!p.superview) continue;
        CGPoint c = [p.superview convertPoint:p.center toView:overlay];
        if (fabs(c.x - target.x) < sq * 0.5 && fabs(c.y - target.y) < sq * 0.5) return YES;
    }
    return NO;
}

static void CAPSendTouch(UIView *overlay, CAPFakeTouch *touch, CAPFakeEvent *event,
                         UITouchPhase phase, CGPoint point) {
    touch.capPrev = touch.capPoint;
    if (phase == UITouchPhaseBegan) touch.capPrev = point;
    touch.capPoint = point;
    touch.capPhase = phase;
    touch.capTime = NSProcessInfo.processInfo.systemUptime;
    event.capTime = touch.capTime;
    SEL sel = phase == UITouchPhaseBegan ? @selector(touchesBegan:withEvent:) :
              phase == UITouchPhaseMoved ? @selector(touchesMoved:withEvent:) :
                                           @selector(touchesEnded:withEvent:);
    ((void (*)(id, SEL, id, id))objc_msgSend)(overlay, sel, event.capTouches, event);
}

// Synthetic drag (or tap-tap) from one point to another on the pieces overlay.
static void CAPSyntheticStep(UIView *overlay, CAPFakeTouch *touch, CAPFakeEvent *event,
                             CGPoint from, CGPoint to, BOOL tapMode, int step) {
    int total = tapMode ? 4 : 6;
    @try {
        if (tapMode) {
            if (step == 0) CAPSendTouch(overlay, touch, event, UITouchPhaseBegan, from);
            else if (step == 1) CAPSendTouch(overlay, touch, event, UITouchPhaseEnded, from);
            else if (step == 2) CAPSendTouch(overlay, touch, event, UITouchPhaseBegan, to);
            else CAPSendTouch(overlay, touch, event, UITouchPhaseEnded, to);
        } else if (step == 0) {
            CAPSendTouch(overlay, touch, event, UITouchPhaseBegan, from);
        } else if (step <= 4) {
            CGFloat t = step / 4.0;
            CAPSendTouch(overlay, touch, event, UITouchPhaseMoved,
                         CGPointMake(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t));
        } else {
            CAPSendTouch(overlay, touch, event, UITouchPhaseEnded, to);
        }
    } @catch (NSException *e) {
        NSLog(@"[CAP-AP] touch exception %@", e);
    }
    if (step + 1 >= total) return;
    int64_t gap = tapMode ? (step + 1 == 2 ? 150 : 16) : 45;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, gap * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
        CAPSyntheticStep(overlay, touch, event, from, to, tapMode, step + 1);
    });
}

static void CAPSyntheticMove(UIView *overlay, CGPoint from, CGPoint to, BOOL tapMode) {
    CAPFakeTouch *touch = [[CAPFakeTouch alloc] init];
    CAPFakeEvent *event = [[CAPFakeEvent alloc] init];
    touch.capView = overlay;
    event.capTouches = [NSSet setWithObject:touch];
    CAPSyntheticStep(overlay, touch, event, from, to, tapMode, 0);
}

// On-screen Auto-Play status (no log access on a sideloaded phone).
static void CAPToast(NSString *text) {
    NSLog(@"[CAP-AP] %@", text);
    dispatch_async(dispatch_get_main_queue(), ^{
        UIWindow *host = nil;
        for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
            if (![scene isKindOfClass:UIWindowScene.class] ||
                scene.activationState != UISceneActivationStateForegroundActive) continue;
            for (UIWindow *w in ((UIWindowScene *)scene).windows) if (w.isKeyWindow) host = w;
            if (!host) host = ((UIWindowScene *)scene).windows.firstObject;
        }
        if (!host) return;
        UILabel *label = (UILabel *)[host viewWithTag:0xCA9];
        if (!label) {
            label = [[UILabel alloc] init];
            label.tag = 0xCA9;
            label.numberOfLines = 3;
            label.font = [UIFont boldSystemFontOfSize:12];
            label.textColor = UIColor.whiteColor;
            label.backgroundColor = [UIColor colorWithWhite:0 alpha:0.75];
            label.layer.cornerRadius = 6;
            label.clipsToBounds = YES;
            label.textAlignment = NSTextAlignmentCenter;
            label.userInteractionEnabled = NO;
            [host addSubview:label];
        }
        label.text = [@"AP: " stringByAppendingString:text];
        CGFloat top = host.safeAreaInsets.top + 44;
        label.frame = CGRectMake(8, top, host.bounds.size.width - 16, 44);
        [host bringSubviewToFront:label];
        label.alpha = 1;
        static NSInteger generation = 0;
        NSInteger mine = ++generation;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 6 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
            if (mine == generation) [label removeFromSuperview];
        });
    });
}

static NSString *CAPDeepVCName(UIViewController *vc, int depth) {
    if (!vc || depth > 12) return @"?";
    UIViewController *next = vc.presentedViewController;
    if (!next && [vc isKindOfClass:UINavigationController.class]) next = ((UINavigationController *)vc).topViewController;
    if (!next && [vc isKindOfClass:UITabBarController.class]) next = ((UITabBarController *)vc).selectedViewController;
    if (!next) next = vc.childViewControllers.lastObject;
    return next ? CAPDeepVCName(next, depth + 1) : NSStringFromClass(vc.class);
}

static BOOL CAPAutoPlaySwiftBot(NSString *uci, NSString *fen) {
    Class overlayClass = NSClassFromString(@"_TtC15SwiftChessBoard28PiecesOverlayWithTouchEvents");
    Class pieceClass = NSClassFromString(@"_TtC15SwiftChessBoard9PieceView");
    if (!overlayClass) { CAPToast(@"board class missing"); return NO; }
    UIView *overlay = nil;
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class] ||
            scene.activationState != UISceneActivationStateForegroundActive) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            overlay = CAPFindOverlay(window, overlayClass);
            if (overlay) break;
        }
        if (overlay) break;
    }
    if (!overlay) { CAPToast(@"no board overlay found"); return NO; }
    if (!CAPIsBotGameScreen(overlay)) { CAPToast([NSString stringWithFormat:@"not bot screen: %@", CAPDeepVCName(overlay.window.rootViewController, 0)]); return NO; }

    int sf = [uci characterAtIndex:0] - 'a', sr = [uci characterAtIndex:1] - '1';
    int df = [uci characterAtIndex:2] - 'a', dr = [uci characterAtIndex:3] - '1';
    BOOL whiteToMove = [fen containsString:@" w "];
    NSMutableArray *pieces = [NSMutableArray array];
    if (pieceClass) CAPCollectPieces(overlay.superview ?: overlay, pieceClass, pieces);

    BOOL flipped = !whiteToMove; // fallback: user is the side to move
    BOOL decided = NO;
    if (pieces.count) {
        uint64_t fenMask = CAPFenMask(fen);
        BOOL m0 = CAPPieceMask(pieces, overlay, NO) == fenMask;
        BOOL m1 = CAPPieceMask(pieces, overlay, YES) == fenMask;
        if (m0 != m1) { flipped = m1; decided = YES; }
    }
    if (decided && whiteToMove == flipped) {
        CAPToast(@"not your turn on this board");
        return NO;
    }
    CGFloat size = overlay.bounds.size.width;
    CGPoint from = CAPSquareCenter(size, sf, sr, flipped);
    CGPoint to = CAPSquareCenter(size, df, dr, flipped);
    CAPToast([NSString stringWithFormat:@"dragging %@ (flip %d, pieces %lu)", uci, flipped, (unsigned long)pieces.count]);
    // The board's own touch handlers implement drag (began on piece, moved, ended
    // on target) via locationInView: only, so drag goes first; taps are the fallback.
    CAPSyntheticMove(overlay, from, to, NO);

    __weak UIView *weakOverlay = overlay;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 900 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
        UIView *o = weakOverlay;
        if (!o.window || !pieceClass) return;
        NSMutableArray *now = [NSMutableArray array];
        CAPCollectPieces(o.superview ?: o, pieceClass, now);
        BOOL stillThere = CAPPieceAt(now, o, sf, sr, flipped);
        CAPToast(stillThere ? @"drag ignored, trying taps" : @"drag worked");
        if (stillThere) CAPSyntheticMove(o, from, to, YES);
    });
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2200 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
        UIView *o = weakOverlay;
        if (!o.window || !pieceClass) return;
        NSMutableArray *now = [NSMutableArray array];
        CAPCollectPieces(o.superview ?: o, pieceClass, now);
        CAPToast([NSString stringWithFormat:@"%@ %@", uci, CAPPieceAt(now, o, sf, sr, flipped) ? @"NOT accepted" : @"accepted"]);
    });
    return YES;
}

static BOOL CAPAutoPlay(NSString *uci, NSString *fen, CAPMoveSource source) {
    if (![[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey] ||
        !CAPSourceEnabled(source) || [lastPlayedFEN isEqualToString:fen]) {
        NSLog(@"[ChessAssistantPro] Auto-Play [selection] disabled, wrong engine, or position already attempted");
        return NO;
    }
    int sf = [uci characterAtIndex:0] - 'a', sr = [uci characterAtIndex:1] - '1';
    int df = [uci characterAtIndex:2] - 'a', dr = [uci characterAtIndex:3] - '1';
    @try {
        UIView *board = CAPVisibleBoardForFEN(fen);
        if (!board) {
            // Swift board: only supported for computer (bot) games.
            if (CAPAutoPlaySwiftBot(uci, fen)) { lastPlayedFEN = [fen copy]; return YES; }
            return NO;
        }
        BOOL whiteToMove = [fen containsString:@" w "];
        SEL permission = whiteToMove ? @selector(allowUserToMakeWhiteMoves)
                                     : @selector(allowUserToMakeBlackMoves);
        if (![board respondsToSelector:permission] ||
            !((BOOL (*)(id, SEL))objc_msgSend)(board, permission)) {
            NSLog(@"[ChessAssistantPro] Auto-Play: player cannot move on this board");
            return NO;
        }
        id game = ((id (*)(id, SEL))objc_msgSend)(board, @selector(chessGame));
        SEL validMove = @selector(isValidMoveFromSourceRank:SourceFile:DestRank:DestFile:);
        SEL initializer = @selector(initSR:SF:DR:DF:PP:);
        Class moveClass = NSClassFromString(@"CHMoveSourceDestination");
        if (!game || ![game respondsToSelector:validMove] || !moveClass ||
            ![moveClass instancesRespondToSelector:initializer]) return NO;
        // CHMoveSourceDestination's lanMove implementation adds 'a'/'1' to its
        // stored file/rank, proving this app uses zero-based square values.
        char sourceRank = (char)sr, sourceFile = (char)sf;
        char destRank = (char)dr, destFile = (char)df;
        char promotion = uci.length == 5 ? (char)[uci characterAtIndex:4] : -1;
        if (promotion != -1 && whiteToMove) promotion = (char)toupper(promotion);
        id candidate = ((id (*)(id, SEL, char, char, char, char, char))objc_msgSend)
            ([moveClass alloc], initializer, sourceRank, sourceFile, destRank, destFile, promotion);
        if (!candidate || ![candidate respondsToSelector:@selector(lanMove)] ||
            ![CAPNormalizeMove(((id (*)(id, SEL))objc_msgSend)(candidate, @selector(lanMove)))
                isEqualToString:uci]) {
            NSLog(@"[ChessAssistantPro] Auto-Play: move encoding mismatch for %@", uci);
            return NO;
        }
        if (!((BOOL (*)(id, SEL, char, char, char, char))objc_msgSend)
              (game, validMove, sourceRank, sourceFile, destRank, destFile)) {
            NSLog(@"[ChessAssistantPro] Auto-Play: %@ is not legal in live board", uci);
            return NO;
        }
        NSString *before = CAPPositionKey(fen);
        NSLog(@"[ChessAssistantPro] Auto-Play [direct] invoking CHBoardView for %@", uci);
        ((void (*)(id, SEL, id, BOOL))objc_msgSend)
            (board, @selector(makeMove:throughTouches:), candidate, NO);
        lastPlayedFEN = [fen copy];
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 400 * NSEC_PER_MSEC),
                       dispatch_get_main_queue(), ^{
            NSString *after = nil;
            id currentGame = board.window ?
                ((id (*)(id, SEL))objc_msgSend)(board, @selector(chessGame)) : nil;
            if (currentGame && [currentGame respondsToSelector:@selector(fenForCurrentPosition:flipped:)])
                after = ((id (*)(id, SEL, BOOL, BOOL))objc_msgSend)
                    (currentGame, @selector(fenForCurrentPosition:flipped:), YES, NO);
            if (!after && board.window &&
                [board respondsToSelector:@selector(fenForCurrentPosition:flipped:)])
                after = ((id (*)(id, SEL, BOOL, BOOL))objc_msgSend)
                    (board, @selector(fenForCurrentPosition:flipped:), YES, NO);
            NSString *position = [after isKindOfClass:NSString.class] ? CAPPositionKey(after) : nil;
            NSLog(@"[ChessAssistantPro] Auto-Play [verify] %@ %@", uci,
                  !position ? @"position unavailable" :
                  ([position isEqualToString:before] ? @"unchanged" : @"advanced"));
        });
        return YES;
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] Auto-Play skipped: %@", exception);
    }
    return NO;
}

static void CAPScheduleAutoPlay(NSString *move, NSString *fen, CAPMoveSource source) {
    NSString *uci = CAPNormalizeMove(move);
    if (!fen.length || (uci.length != 4 && uci.length != 5)) {
        if ([[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey])
            NSLog(@"[ChessAssistantPro] Auto-Play [selection] missing FEN or invalid engine move %@", move);
        return;
    }
    if ([uci characterAtIndex:0] < 'a' || [uci characterAtIndex:0] > 'h' ||
        [uci characterAtIndex:2] < 'a' || [uci characterAtIndex:2] > 'h' ||
        [uci characterAtIndex:1] < '1' || [uci characterAtIndex:1] > '8' ||
        [uci characterAtIndex:3] < '1' || [uci characterAtIndex:3] > '8' ||
        (uci.length == 5 && [@"qrbn" rangeOfString:[uci substringFromIndex:4]].location == NSNotFound)) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (![[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey] ||
            !CAPSourceEnabled(source) || [lastPlayedFEN isEqualToString:fen] ||
            [lastScheduledFEN isEqualToString:fen] ||
            ![CAPPositionKey(fen) isEqualToString:lastObservedPosition]) return;
        lastScheduledFEN = [fen copy];
        uint64_t generation = ++autoPlayGeneration;
        double minimum = CAPDelay(kAutoPlayMinDelayKey, 1.0);
        double maximum = fmax(minimum, CAPDelay(kAutoPlayMaxDelayKey, 3.0));
        double fraction = (double)arc4random() / (double)UINT32_MAX;
        double delay = minimum + fraction * (maximum - minimum);
        NSLog(@"[ChessAssistantPro] Auto-Play queued %@ in %.3f s", uci, delay);
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)),
                       dispatch_get_main_queue(), ^{
            if (generation != autoPlayGeneration) return;
            if (!CAPAutoPlay(uci, fen, source)) lastScheduledFEN = nil;
        });
    });
}

static void CAPScheduleCachedMove(void) {
    if (![[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey]) return;
    CAPAutoEngine selected = CAPSelectedAutoEngine();
    if (selected == CAPAutoEngineMaia)
        CAPScheduleAutoPlay(cachedMaiaMove, cachedMaiaFEN, CAPMoveSourceMaia);
    else if (cachedGodChoice == (selected == CAPAutoEngineTriumviratus ? 1 : 0))
        CAPScheduleAutoPlay(cachedGodMove, cachedGodFEN, CAPMoveSourceGod);
}

static void CAPRememberMove(NSString *move, NSString *fen, CAPMoveSource source) {
    if (!move.length || !fen.length) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (![CAPPositionKey(fen) isEqualToString:lastObservedPosition]) return;
        if (source == CAPMoveSourceMaia) {
            cachedMaiaMove = [move copy]; cachedMaiaFEN = [fen copy];
        } else {
            cachedGodMove = [move copy]; cachedGodFEN = [fen copy];
            cachedGodChoice = [[NSUserDefaults standardUserDefaults] integerForKey:kGodEngineKey] == 1 ? 1 : 0;
        }
        CAPScheduleAutoPlay(move, fen, source);
    });
}

static void CAPMaiaGo(const char *fen, uintptr_t selfElo, uintptr_t oppoElo,
                      MaiaCallback callback) {
    if (!originalMaiaGo) return;
    size_t length = fen ? strnlen(fen, 256) : 0;
    NSString *position = length > 0 && length < 256 ?
        [[NSString alloc] initWithBytes:fen length:length encoding:NSUTF8StringEncoding] : nil;
    CAPObservePosition(position);
    MaiaCallback forwarded = ^(MaiaResult result) {
        if (callback) callback(result);
        if (!position) return;
        size_t moveLength = strnlen(result.move, sizeof(result.move));
        if (moveLength >= 4 && moveLength <= 5) {
            NSString *move = [[NSString alloc] initWithBytes:result.move length:moveLength
                                                    encoding:NSASCIIStringEncoding];
            CAPRememberMove(move, position, CAPMoveSourceMaia);
        }
    };
    originalMaiaGo(fen, selfElo, oppoElo, [forwarded copy]);
}

// EngineGo's callback ABI is recovered from the injected Chess.dylib. Returning
// this result lets its existing CHPad/board path draw the normal engine arrow.
struct EngineLine { char move[8]; bool flag1; bool flag2; int value; };
static_assert(sizeof(EngineLine) == 16, "Engine callback ABI changed");
typedef void (^EngineCallback)(const EngineLine *, int);
typedef void (*EngineGoFn)(const char *, uintptr_t, uintptr_t, uintptr_t, EngineCallback);
static EngineGoFn originalEngineGo = nullptr;
static std::atomic<uint64_t> engineGeneration{0};

static BOOL CAPWrite(int fd, const std::string &command) {
    const char *data = command.data();
    size_t remaining = command.size();
    while (remaining) {
        ssize_t written = write(fd, data, remaining);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return NO;
        data += written;
        remaining -= (size_t)written;
    }
    return YES;
}

static BOOL CAPReadLine(int fd, std::string &line, CFAbsoluteTime deadline) {
    line.clear();
    while (line.size() < 4096 && CFAbsoluteTimeGetCurrent() < deadline) {
        int timeout = (int)((deadline - CFAbsoluteTimeGetCurrent()) * 1000);
        struct pollfd item = {fd, POLLIN, 0};
        int ready = poll(&item, 1, MAX(timeout, 1));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0 || (item.revents & (POLLERR | POLLNVAL))) return NO;
        char character;
        ssize_t count = read(fd, &character, 1);
        if (count <= 0) return NO;
        if (character == '\n') return YES;
        if (character != '\r') line.push_back(character);
    }
    return NO;
}

static BOOL CAPWaitForLine(int fd, const char *expected, CFAbsoluteTime deadline) {
    std::string line;
    while (CAPReadLine(fd, line, deadline)) {
        if (line == expected) return YES;
    }
    return NO;
}

static BOOL CAPUciMove(const std::string &move) {
    if (move.size() != 4 && move.size() != 5) return NO;
    if (move[0] < 'a' || move[0] > 'h' || move[2] < 'a' || move[2] > 'h' ||
        move[1] < '1' || move[1] > '8' || move[3] < '1' || move[3] > '8') return NO;
    return move.size() == 4 || move[4] == 'q' || move[4] == 'r' ||
           move[4] == 'b' || move[4] == 'n';
}

static int CAPRequestedStockfishElo(void) {
    id saved = [[NSUserDefaults standardUserDefaults] objectForKey:kStrengthEloKey];
    // The original panel defaults to its maximum when there is no saved value.
    return saved && [saved respondsToSelector:@selector(integerValue)] ?
        (int)MAX(600, MIN(10000, [saved integerValue])) : 3200;
}

// Low-ELO move selection and FEN feature analysis.
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

static double CAPStockfishSkillForElo(int elo) {
    if (elo < 1320) {
        NSNumber *skill = HEDescribeElo(elo)[@"skill"];
        return skill.doubleValue;
    }
    if (elo >= 3190) return 20;
    double e = (double)(elo - 1320) / (3190 - 1320);
    return fmax(0.0, fmin(19.0,
        (((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438)));
}

// FEN fullmove starts at 1. Search heuristics need half-moves played so far.
static int CAPFenPly(const char *fen) {
    if (!fen) return -1;
    std::istringstream fields(fen);
    std::string placement, side, castling, ep;
    int halfmove = 0, fullmove = 0;
    if (!(fields >> placement >> side >> castling >> ep >> halfmove >> fullmove) ||
        (side != "w" && side != "b") || fullmove < 1 || fullmove > 10000)
        return -1;
    return 2 * (fullmove - 1) + (side == "b" ? 1 : 0);
}

static BOOL CAPExternalEngineMove(const char *path, const char *fen,
                                  BOOL stockfish19, int requestedElo,
                                  std::string &bestMove) {
    if (!path || !fen || access(path, X_OK) != 0) return NO;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    // Accessed only from the serial God Engine queue. Keep Stockfish warm so
    // later positions do not pay process launch and NNUE initialization costs.
    static int cachedInput = -1, cachedOutput = -1, cachedThreads = 0;
    static pid_t cachedPid = 0;
    static std::string humanLastFen, humanLastMove;
    static int humanLastPly = -1, humanLastElo = -1;
    if (stockfish19 && cachedPid > 0) {
        int status = 0;
        pid_t exited;
        do { exited = waitpid(cachedPid, &status, WNOHANG); }
        while (exited < 0 && errno == EINTR);
        if (exited != 0) {
            close(cachedInput); close(cachedOutput);
            cachedInput = cachedOutput = -1;
            cachedPid = 0;
            cachedThreads = 0;
        }
    }
    int input[2] = {-1, -1}, output[2] = {-1, -1};
    pid_t pid = 0;
    BOOL reused = stockfish19 && cachedPid > 0;
    if (reused) {
        input[1] = cachedInput;
        output[0] = cachedOutput;
        pid = cachedPid;
    } else {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, input) != 0) return NO;
        if (pipe(output) != 0) { close(input[0]); close(input[1]); return NO; }
        int noSigPipe = 1;
        if (setsockopt(input[1], SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, sizeof(noSigPipe)) != 0) {
            close(input[0]); close(input[1]); close(output[0]); close(output[1]);
            return NO;
        }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&actions, input[1]);
        posix_spawn_file_actions_addclose(&actions, output[0]);
        char *const arguments[] = {const_cast<char *>(path), nullptr};
        int status = posix_spawn(&pid, path, &actions, nullptr, arguments, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(input[0]); close(output[1]);
        if (status != 0) { close(input[1]); close(output[0]); return NO; }
    }
    BOOL success = NO;
    std::string line;
    BOOL lowElo = stockfish19 && requestedElo < 1320;
    int ply = lowElo ? CAPFenPly(fen) : -1;
    NSString *humanGo = nil;
    if (lowElo) {
        BOOL newGame = humanLastPly < 0 || ply < humanLastPly ||
            (ply <= 1 && humanLastPly > 1);
        if (newGame) {
            HENewGame(strstr(fen, " b ") == nullptr, requestedElo);
            humanLastFen.clear(); humanLastMove.clear();
            NSLog(@"[ChessAssistantPro] Human band new game, ply=%d", ply);
        } else if (ply == humanLastPly + 2 && !humanLastFen.empty() &&
                   !humanLastMove.empty()) {
            NSString *opponent = HEInferOpponentMove(humanLastFen.c_str(),
                humanLastMove.c_str(), fen);
            if (opponent) {
                HENoteOpponentMove(opponent.UTF8String);
                NSLog(@"[ChessAssistantPro] Human band opponent move %@", opponent);
            } else {
                HENoteOpponentMove(nullptr);
                NSLog(@"[ChessAssistantPro] Human band opponent move unavailable from FEN transition");
            }
        } else if (ply > humanLastPly && ply != humanLastPly + 2) {
            HENoteOpponentMove(nullptr);
        }
    }
    CFAbsoluteTime deadline = CFAbsoluteTimeGetCurrent() + 5.0;
    if (!reused) {
        if (!CAPWrite(input[1], "uci\n") || !CAPWaitForLine(output[0], "uciok", deadline)) goto done;
        if (stockfish19 && !CAPWrite(input[1], "setoption name Hash value 32\n")) goto done;
    }
    if (stockfish19) {
        // Reapply on every position: the engine process is reused after slider changes.
        int uciElo = MAX(1320, MIN(3190, requestedElo));
        if (!lowElo && cachedThreads != 2) {
            if (!CAPWrite(input[1], "setoption name Threads value 2\n")) goto done;
            cachedThreads = 2;
        }
        if (lowElo) {
            HEParams params = HEParamsFor(requestedElo);
            if (cachedThreads != 1) {
                if (!CAPWrite(input[1], "setoption name Threads value 1\n")) goto done;
                cachedThreads = 1;
            }
            if (!CAPWrite(input[1], "setoption name MultiPV value " +
                                    std::to_string(params.multipv) + "\n") ||
                !CAPWrite(input[1], "setoption name Skill Level value " +
                                    std::to_string(params.skill) + "\n") ||
                !CAPWrite(input[1], "setoption name UCI_LimitStrength value false\n") ||
                !CAPWrite(input[1], "setoption name UCI_Elo value 1320\n")) goto done;
            int depth = params.depth + (int)arc4random_uniform(3);
            int moveTime = 110 + (int)arc4random_uniform(140);
            humanGo = [NSString stringWithFormat:@"go depth %d movetime %d",
                       depth, moveTime];
            HEResetLines();
            NSLog(@"[ChessAssistantPro] Stockfish human band ELO=%d ply=%d %@",
                  requestedElo, ply, humanGo);
        } else {
            if (!CAPWrite(input[1], "setoption name MultiPV value 1\n") ||
                !CAPWrite(input[1], "setoption name Skill Level value 20\n") ||
                !CAPWrite(input[1], std::string("setoption name UCI_LimitStrength value ") +
                                    (requestedElo >= 3190 ? "false\n" : "true\n")) ||
                !CAPWrite(input[1], "setoption name UCI_Elo value " +
                                    std::to_string(uciElo) + "\n")) goto done;
            NSLog(@"[ChessAssistantPro] Stockfish strength slider=%d UCI_Elo=%d skill=%.1f/20 MultiPV=1",
                  requestedElo, uciElo, CAPStockfishSkillForElo(requestedElo));
        }
    }
    if (!CAPWrite(input[1], "isready\n") || !CAPWaitForLine(output[0], "readyok", deadline)) goto done;
    if (!CAPWrite(input[1], std::string("position fen ") + fen + "\n")) goto done;
    if (!CAPWrite(input[1], lowElo ? std::string(humanGo.UTF8String) + "\n" :
                          (stockfish19 ? "go movetime 200\n" : "go depth 16\n"))) goto done;
    deadline = CFAbsoluteTimeGetCurrent() + (stockfish19 ? 3.0 : 15.0);
    while (CAPReadLine(output[0], line, deadline)) {
        if (line.find("CRITICAL ERROR") != std::string::npos) break;
        if (line.compare(0, 9, "bestmove ") != 0) {
            if (lowElo) HEParseInfo(line.c_str());
            continue;
        }
        size_t end = line.find(' ', 9);
        bestMove = line.substr(9, end == std::string::npos ? end : end - 9);
        success = CAPUciMove(bestMove);
        if (success && lowElo) {
            if (humanLastFen == fen && humanLastElo == requestedElo &&
                CAPUciMove(humanLastMove)) {
                bestMove = humanLastMove;
            } else {
                NSString *chosen = HEChooseMove(fen, MAX(ply, 0), requestedElo);
                if (chosen && CAPUciMove(chosen.UTF8String)) bestMove = chosen.UTF8String;
                humanLastFen = fen;
                humanLastMove = bestMove;
                humanLastPly = ply;
                humanLastElo = requestedElo;
            }
        }
        break;
    }
done:
    if (stockfish19 && success) {
        cachedInput = input[1]; cachedOutput = output[0]; cachedPid = pid;
        NSLog(@"[ChessAssistantPro] Stockfish 19 bestmove %s in %.0f ms (%@)",
              bestMove.c_str(), (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
              reused ? @"warm" : @"cold");
        return YES;
    }
    if (stockfish19) {
        cachedInput = cachedOutput = -1;
        cachedPid = 0;
        cachedThreads = 0;
    }
    close(input[1]); close(output[0]);
    kill(pid, SIGKILL);
    while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    return success;
}

static void CAPEngineGo(const char *fen, uintptr_t arg1, uintptr_t arg2,
                        uintptr_t arg3, EngineCallback callback) {
    if (!originalEngineGo) return;
    uint64_t request = engineGeneration.fetch_add(1) + 1;
    size_t length = fen ? strnlen(fen, 256) : 0;
    NSString *position = length > 0 && length < 256 ?
        [[NSString alloc] initWithBytes:fen length:length encoding:NSUTF8StringEncoding] : nil;
    CAPObservePosition(position);
    EngineCallback originalCallback = [callback copy];
    EngineCallback forwarded = [^(const EngineLine *lines, int count) {
        if (originalCallback) originalCallback(lines, count);
        if (!position || !lines || count <= 0) return;
        size_t moveLength = strnlen(lines[0].move, sizeof(lines[0].move));
        if (moveLength >= 4 && moveLength <= 5) {
            NSString *move = [[NSString alloc] initWithBytes:lines[0].move length:moveLength
                                                    encoding:NSASCIIStringEncoding];
            CAPRememberMove(move, position, CAPMoveSourceGod);
        }
    } copy];
    if (length == 0 || length >= 256 || strpbrk(fen, "\r\n")) {
        originalEngineGo(fen, arg1, arg2, arg3, forwarded);
        return;
    }
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    int requestedElo = CAPRequestedStockfishElo();
    BOOL useTriumviratus = [defaults boolForKey:kAutoPlayKey] &&
        CAPSelectedAutoEngine() != CAPAutoEngineMaia ?
        CAPSelectedAutoEngine() == CAPAutoEngineTriumviratus :
        [defaults integerForKey:kGodEngineKey] == 1;
    if (access(useTriumviratus ? kTriumviratusPath : kStockfish19Path, X_OK) != 0 &&
        access(kStockfish19Path, X_OK) != 0) {
        originalEngineGo(fen, arg1, arg2, arg3, forwarded);
        return;
    }
    if (!position) { originalEngineGo(fen, arg1, arg2, arg3, forwarded); return; }
    static dispatch_queue_t queue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ queue = dispatch_queue_create("com.jerich.chessassistantpro.godengine", DISPATCH_QUEUE_SERIAL); });
    dispatch_async(queue, ^{
        if (engineGeneration.load() != request) return;
        std::string move;
        BOOL found = NO;
        @try {
            if (useTriumviratus)
                found = CAPExternalEngineMove(kTriumviratusPath, position.UTF8String,
                                              NO, requestedElo, move);
            if (!found)
                found = CAPExternalEngineMove(kStockfish19Path, position.UTF8String,
                                              YES, requestedElo, move);
        } @catch (NSException *exception) {
            NSLog(@"[ChessAssistantPro] External engine failed: %@", exception);
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            if (engineGeneration.load() != request) return;
            if (!found) {
                NSLog(@"[ChessAssistantPro] External engine unavailable; using embedded Stockfish 16");
                originalEngineGo(position.UTF8String, arg1, arg2, arg3, forwarded);
                return;
            }
            EngineLine result = {};
            strlcpy(result.move, move.c_str(), sizeof(result.move));
            forwarded(&result, 1);
        });
    });
}

static UIButton *CAPFindPauseButton(UIView *view) {
    if ([view isKindOfClass:UIButton.class]) {
        UIButton *button = (UIButton *)view;
        NSString *title = [button titleForState:UIControlStateNormal] ?: button.titleLabel.text;
        if ([title rangeOfString:@"Pause Assistant" options:NSCaseInsensitiveSearch].location != NSNotFound ||
            [title rangeOfString:@"Enable Assistant" options:NSCaseInsensitiveSearch].location != NSNotFound)
            return button;
    }
    for (UIView *child in view.subviews) {
        UIButton *button = CAPFindPauseButton(child);
        if (button) return button;
    }
    return nil;
}

static char kAutoPlayRowKey, kAutoEngineRowKey, kDelayRowKey,
            kMinDelayFieldKey, kMaxDelayFieldKey;
static char kGodEngineRowKey, kStockfishSkillRowKey, kStockfishSkillLabelKey;

static UISegmentedControl *CAPFindSegment(UIView *view) {
    if ([view isKindOfClass:UISegmentedControl.class]) return (UISegmentedControl *)view;
    for (UIView *child in view.subviews) {
        UISegmentedControl *found = CAPFindSegment(child);
        if (found) return found;
    }
    return nil;
}

static void CAPSyncEngineChoiceUI(id panel) {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    CAPAutoEngine selected = CAPSelectedAutoEngine();
    if (selected != CAPAutoEngineMaia)
        [defaults setInteger:selected == CAPAutoEngineTriumviratus ? 1 : 0 forKey:kGodEngineKey];
    if (![panel isKindOfClass:UIView.class]) return;
    UIView *godRow = objc_getAssociatedObject(panel, &kGodEngineRowKey);
    UISegmentedControl *god = godRow ? CAPFindSegment(godRow) : nil;
    if (god) god.selectedSegmentIndex = selected == CAPAutoEngineTriumviratus ? 1 : 0;
    UIView *autoRow = objc_getAssociatedObject(panel, &kAutoEngineRowKey);
    UISegmentedControl *autoChoice = autoRow ? CAPFindSegment(autoRow) : nil;
    if (autoChoice) autoChoice.selectedSegmentIndex = selected;
    // The original assistant fetches recommendations from its board layout
    // hooks. Refresh after an engine switch so this position gets a new arrow.
    UIView *board = CAPVisibleSwiftBoard();
    if (board) { [board setNeedsLayout]; [board layoutIfNeeded]; }
}

static NSString *CAPDelayText(double seconds) {
    NSMutableString *text = [[NSString stringWithFormat:@"%.6f", seconds] mutableCopy];
    while ([text hasSuffix:@"0"]) [text deleteCharactersInRange:NSMakeRange(text.length - 1, 1)];
    if ([text hasSuffix:@"."]) [text deleteCharactersInRange:NSMakeRange(text.length - 1, 1)];
    return text;
}

static UITextField *CAPDelayField(id panel, NSString *placeholder, double value, NSInteger tag) {
    UITextField *field = [[UITextField alloc] initWithFrame:CGRectZero];
    field.tag = tag;
    field.text = CAPDelayText(value);
    field.placeholder = placeholder;
    field.keyboardType = UIKeyboardTypeDecimalPad;
    field.textAlignment = NSTextAlignmentCenter;
    field.font = [UIFont monospacedDigitSystemFontOfSize:13 weight:UIFontWeightMedium];
    field.textColor = UIColor.labelColor;
    field.backgroundColor = UIColor.tertiarySystemFillColor;
    field.layer.cornerRadius = 7;
    [field.widthAnchor constraintEqualToConstant:64].active = YES;
    [field.heightAnchor constraintEqualToConstant:32].active = YES;
    [field addTarget:panel action:@selector(cap_delayChanged:)
        forControlEvents:UIControlEventEditingDidEnd];
    UIToolbar *toolbar = [[UIToolbar alloc] initWithFrame:CGRectMake(0, 0, 320, 44)];
    UIBarButtonItem *space = [[UIBarButtonItem alloc]
        initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    UIBarButtonItem *done = [[UIBarButtonItem alloc]
        initWithBarButtonSystemItem:UIBarButtonSystemItemDone
        target:panel action:@selector(cap_delayDone:)];
    toolbar.items = @[space, done];
    field.inputAccessoryView = toolbar;
    return field;
}

static void CAPInsertAutoPlay(id panel) {
    if (![panel isKindOfClass:UIView.class]) return;
    UIView *existing = objc_getAssociatedObject(panel, &kAutoPlayRowKey);
    UIView *delayRow = objc_getAssociatedObject(panel, &kDelayRowKey);
    if ([existing isDescendantOfView:(UIView *)panel] &&
        [delayRow isDescendantOfView:(UIView *)panel]) return;
    UIButton *pause = CAPFindPauseButton((UIView *)panel);
    if (!pause) return;
    UIView *child = pause;
    UIStackView *stack = nil;
    while (child.superview) {
        UIView *parent = child.superview;
        if ([parent isKindOfClass:UIStackView.class] &&
            [((UIStackView *)parent).arrangedSubviews containsObject:child]) {
            stack = (UIStackView *)parent;
            break;
        }
        child = parent;
    }
    if (!stack || ![panel respondsToSelector:@selector(switchOn:sel:)] ||
        ![panel respondsToSelector:@selector(rowTitle:control:)]) return;
    NSUInteger index = [stack.arrangedSubviews indexOfObject:child];
    if (index == NSNotFound) return;
    UIView *autoEngineRow = objc_getAssociatedObject(panel, &kAutoEngineRowKey);
    if (![autoEngineRow isDescendantOfView:(UIView *)panel]) {
        UISegmentedControl *choice = [[UISegmentedControl alloc]
            initWithItems:@[@"Maia", @"SF 19", @"Trium"]];
        choice.selectedSegmentIndex = CAPSelectedAutoEngine();
        [choice addTarget:panel action:@selector(cap_autoEngineChanged:)
             forControlEvents:UIControlEventValueChanged];
        if ([panel respondsToSelector:@selector(styleSeg:)])
            ((void (*)(id, SEL, id))objc_msgSend)(panel, @selector(styleSeg:), choice);
        UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
            (panel, @selector(rowTitle:control:), @"Auto Engine", choice);
        if (![row isKindOfClass:UIView.class]) return;
        [stack insertArrangedSubview:row atIndex:index++];
        objc_setAssociatedObject(panel, &kAutoEngineRowKey, row, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    if (![existing isDescendantOfView:(UIView *)panel]) {
        UISwitch *toggle = ((id (*)(id, SEL, BOOL, SEL))objc_msgSend)
            (panel, @selector(switchOn:sel:),
             [[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey],
             @selector(cap_autoPlayChanged:));
        UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
            (panel, @selector(rowTitle:control:), @"Auto-Play", toggle);
        if (![row isKindOfClass:UIView.class]) return;
        [stack insertArrangedSubview:row atIndex:index++];
        objc_setAssociatedObject(panel, &kAutoPlayRowKey, row, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    if (![delayRow isDescendantOfView:(UIView *)panel]) {
        UITextField *minimum = CAPDelayField(panel, @"Min", CAPDelay(kAutoPlayMinDelayKey, 1.0), 1);
        UITextField *maximum = CAPDelayField(panel, @"Max", CAPDelay(kAutoPlayMaxDelayKey, 3.0), 2);
        UILabel *separator = [[UILabel alloc] initWithFrame:CGRectZero];
        separator.text = @"–";
        separator.textColor = UIColor.secondaryLabelColor;
        UIStackView *range = [[UIStackView alloc]
            initWithArrangedSubviews:@[minimum, separator, maximum]];
        range.axis = UILayoutConstraintAxisHorizontal;
        range.alignment = UIStackViewAlignmentCenter;
        range.spacing = 5;
        UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
            (panel, @selector(rowTitle:control:), @"Delay (s)", range);
        if (![row isKindOfClass:UIView.class]) return;
        [stack insertArrangedSubview:row atIndex:index];
        objc_setAssociatedObject(panel, &kDelayRowKey, row, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        objc_setAssociatedObject(panel, &kMinDelayFieldKey, minimum, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        objc_setAssociatedObject(panel, &kMaxDelayFieldKey, maximum, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
}

static void CAPInsertGodEngine(id panel) {
    if (![panel isKindOfClass:UIView.class]) return;
    UIView *existing = objc_getAssociatedObject(panel, &kGodEngineRowKey);
    if ([existing isDescendantOfView:(UIView *)panel]) return;
    UIButton *pause = CAPFindPauseButton((UIView *)panel);
    if (!pause) return;
    UIView *child = pause;
    UIStackView *stack = nil;
    while (child.superview) {
        UIView *parent = child.superview;
        if ([parent isKindOfClass:UIStackView.class] &&
            [((UIStackView *)parent).arrangedSubviews containsObject:child]) {
            stack = (UIStackView *)parent;
            break;
        }
        child = parent;
    }
    if (!stack || ![panel respondsToSelector:@selector(rowTitle:control:)]) return;
    UISegmentedControl *choice = [[UISegmentedControl alloc]
        initWithItems:@[@"Stockfish 19", @"Triumviratus"]];
    choice.selectedSegmentIndex = [[NSUserDefaults standardUserDefaults] integerForKey:kGodEngineKey] == 1 ? 1 : 0;
    [choice addTarget:panel action:@selector(cap_godEngineChanged:)
       forControlEvents:UIControlEventValueChanged];
    if ([panel respondsToSelector:@selector(styleSeg:)])
        ((void (*)(id, SEL, id))objc_msgSend)(panel, @selector(styleSeg:), choice);
    UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
        (panel, @selector(rowTitle:control:), @"God Engine", choice);
    if (![row isKindOfClass:UIView.class]) return;
    NSUInteger index = [stack.arrangedSubviews indexOfObject:child];
    if (index == NSNotFound) return;
    [stack insertArrangedSubview:row atIndex:index];
    objc_setAssociatedObject(panel, &kGodEngineRowKey, row, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static void CAPUpdateStockfishSkillUI(id panel, UISlider *slider) {
    int elo = CAPRequestedStockfishElo();
    // Keep the number shown during a drag and the engine's next UCI request in sync.
    if ([slider isKindOfClass:UISlider.class] && slider.value >= 600 &&
        slider.value <= 10000 && slider.maximumValue >= 3190) {
        elo = (int)lroundf(slider.value);
        [[NSUserDefaults standardUserDefaults] setInteger:elo forKey:kStrengthEloKey];
    }
    UILabel *label = objc_getAssociatedObject(panel, &kStockfishSkillLabelKey);
    if (!label) return;
    if (elo >= 3190) {
        label.text = @"Max · Skill 20/20";
        label.accessibilityLabel = @"Stockfish 19 full strength, skill level 20 of 20, no fixed Elo rating";
    } else if (elo >= 1320) {
        label.text = [NSString stringWithFormat:@"%d · Skill ~%.1f/20",
                      elo, CAPStockfishSkillForElo(elo)];
        label.accessibilityLabel = [NSString stringWithFormat:
            @"Stockfish 19 ELO limit %d, approximate effective skill %.1f of 20",
            elo, CAPStockfishSkillForElo(elo)];
    } else {
        label.text = [NSString stringWithFormat:@"~%d · Skill %.0f/20 + variety",
                      elo, CAPStockfishSkillForElo(elo)];
        label.accessibilityLabel = [NSString stringWithFormat:
            @"Stockfish 19 uncalibrated low strength %d, skill %.0f of 20 with varied legal moves",
            elo, CAPStockfishSkillForElo(elo)];
    }
}

static void CAPInsertStockfishSkill(id panel) {
    if (![panel isKindOfClass:UIView.class] ||
        ![panel respondsToSelector:@selector(rowTitle:control:)]) return;
    UIView *godRow = objc_getAssociatedObject(panel, &kGodEngineRowKey);
    if (![godRow.superview isKindOfClass:UIStackView.class]) return;
    UIStackView *stack = (UIStackView *)godRow.superview;
    UIView *existing = objc_getAssociatedObject(panel, &kStockfishSkillRowKey);
    if (![existing isDescendantOfView:(UIView *)panel]) {
        UILabel *label = [[UILabel alloc] initWithFrame:CGRectZero];
        label.font = [UIFont monospacedDigitSystemFontOfSize:12 weight:UIFontWeightMedium];
        label.textColor = UIColor.secondaryLabelColor;
        label.textAlignment = NSTextAlignmentRight;
        label.adjustsFontSizeToFitWidth = YES;
        label.minimumScaleFactor = 0.8;
        UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
            (panel, @selector(rowTitle:control:), @"Stockfish Skill", label);
        NSUInteger index = [stack.arrangedSubviews indexOfObject:godRow];
        if (![row isKindOfClass:UIView.class] || index == NSNotFound) return;
        [stack insertArrangedSubview:row atIndex:index + 1];
        objc_setAssociatedObject(panel, &kStockfishSkillRowKey, row,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        objc_setAssociatedObject(panel, &kStockfishSkillLabelKey, label,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    CAPUpdateStockfishSkillUI(panel, nil);
}

static char kDumpGestureKey;
static void CAPInstallDumpGesture(id panel) {
    if (![panel isKindOfClass:UIView.class] ||
        objc_getAssociatedObject(panel, &kDumpGestureKey)) return;
    UILongPressGestureRecognizer *gesture = [[UILongPressGestureRecognizer alloc]
        initWithTarget:panel action:@selector(cap_dumpLongPress:)];
    gesture.minimumPressDuration = 0.8;
    gesture.cancelsTouchesInView = NO;
    [(UIView *)panel addGestureRecognizer:gesture];
    objc_setAssociatedObject(panel, &kDumpGestureKey, gesture, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

%hook CHSettingsPanel
- (void)build {
    %orig;
    CAPInsertGodEngine(self);
    CAPInsertStockfishSkill(self);
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
}
- (void)populate {
    %orig;
    CAPInsertGodEngine(self);
    CAPInsertStockfishSkill(self);
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
}
- (void)eloSlide:(UISlider *)sender {
    %orig;
    CAPUpdateStockfishSkillUI(self, sender);
}
- (void)eloDone:(UISlider *)sender {
    %orig;
    CAPUpdateStockfishSkillUI(self, sender);
}
%new
- (void)cap_dumpLongPress:(UILongPressGestureRecognizer *)gesture {
    if (gesture.state != UIGestureRecognizerStateBegan) return;
    UIView *board = CAPVisibleSwiftBoard();
    if (board) {
        @try { CAPDumpSwiftBoard(board, YES); }
        @catch (NSException *exception) {
            CAPDiscLog(@"DUMP forced failed: %@ %@", exception.name, exception.reason);
        }
    }
    else CAPDiscLog(@"LONG PRESS no visible Swift board");
}
- (void)swMaia:(UISwitch *)sender {
    %orig;
    if ([[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey]) {
        CAPAutoEngine selected = sender.on ? CAPAutoEngineMaia :
            ([[NSUserDefaults standardUserDefaults] integerForKey:kGodEngineKey] == 1 ?
             CAPAutoEngineTriumviratus : CAPAutoEngineStockfish19);
        [[NSUserDefaults standardUserDefaults] setInteger:selected forKey:kAutoEngineKey];
        CAPSyncEngineChoiceUI(self);
        ++autoPlayGeneration;
        lastScheduledFEN = nil;
        lastPlayedFEN = nil;
        CAPScheduleCachedMove();
    }
}
%new
- (void)cap_godEngineChanged:(UISegmentedControl *)sender {
    [[NSUserDefaults standardUserDefaults] setInteger:(sender.selectedSegmentIndex == 1 ? 1 : 0)
                                               forKey:kGodEngineKey];
    if ([[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey]) {
        [[NSUserDefaults standardUserDefaults] setInteger:
            (sender.selectedSegmentIndex == 1 ? CAPAutoEngineTriumviratus : CAPAutoEngineStockfish19)
                                              forKey:kAutoEngineKey];
        CAPSyncEngineChoiceUI(self);
    }
    ++autoPlayGeneration;
    lastScheduledFEN = nil;
    lastPlayedFEN = nil;
    CAPScheduleCachedMove();
}
%new
- (void)cap_autoEngineChanged:(UISegmentedControl *)sender {
    if (sender.selectedSegmentIndex < 0 || sender.selectedSegmentIndex > 2) return;
    [[NSUserDefaults standardUserDefaults] setInteger:sender.selectedSegmentIndex forKey:kAutoEngineKey];
    CAPSyncEngineChoiceUI(self);
    ++autoPlayGeneration;
    lastScheduledFEN = nil;
    lastPlayedFEN = nil;
    CAPScheduleCachedMove();
}
%new
- (void)cap_autoPlayChanged:(UISwitch *)sender {
    [[NSUserDefaults standardUserDefaults] setBool:sender.on forKey:kAutoPlayKey];
    ++autoPlayGeneration;
    lastScheduledFEN = nil;
    lastPlayedFEN = nil;
    if (sender.on) {
        CAPSyncEngineChoiceUI(self);
        CAPScheduleCachedMove();
    }
}
%new
- (void)cap_delayDone:(id)sender {
    [(UIView *)self endEditing:YES];
}
%new
- (void)cap_delayChanged:(UITextField *)sender {
    NSString *raw = [sender.text stringByReplacingOccurrencesOfString:@"," withString:@"."];
    NSScanner *scanner = [NSScanner scannerWithString:raw ?: @""];
    scanner.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    double value = 0;
    if (![scanner scanDouble:&value] || !scanner.isAtEnd || !isfinite(value)) {
        sender.text = CAPDelayText(CAPDelay(sender.tag == 1 ? kAutoPlayMinDelayKey :
                                           kAutoPlayMaxDelayKey, sender.tag == 1 ? 1.0 : 3.0));
        return;
    }
    value = fmin(3.0, fmax(0.000001, value));
    UITextField *minimum = objc_getAssociatedObject(self, &kMinDelayFieldKey);
    UITextField *maximum = objc_getAssociatedObject(self, &kMaxDelayFieldKey);
    double low = sender.tag == 1 ? value : CAPDelay(kAutoPlayMinDelayKey, 1.0);
    double high = sender.tag == 2 ? value : CAPDelay(kAutoPlayMaxDelayKey, 3.0);
    if (low > high) {
        if (sender.tag == 1) high = low;
        else low = high;
    }
    [[NSUserDefaults standardUserDefaults] setDouble:low forKey:kAutoPlayMinDelayKey];
    [[NSUserDefaults standardUserDefaults] setDouble:high forKey:kAutoPlayMaxDelayKey];
    minimum.text = CAPDelayText(low);
    maximum.text = CAPDelayText(high);
}
%end

// Only recolor shapes whose setter is called by the injected original
// Chess.dylib. This keeps system, board, and bottom-sheet shape layers intact.
typedef void (*CAPColorIMP)(CAShapeLayer *, SEL, CGColorRef);
static CAPColorIMP originalStrokeColor = nullptr, originalFillColor = nullptr;
static BOOL CAPOriginalArrowCaller(void *caller) {
    Dl_info info = {};
    return caller && dladdr(caller, &info) && info.dli_fname &&
           strstr(info.dli_fname, "/Chess.dylib");
}
static CGColorRef CAPMidnightBlue(void) {
    return [UIColor colorWithRed:25.0/255.0 green:25.0/255.0 blue:112.0/255.0 alpha:1].CGColor;
}
static void CAPSetStrokeColor(CAShapeLayer *layer, SEL selector, CGColorRef color) {
    originalStrokeColor(layer, selector,
        CAPOriginalArrowCaller(__builtin_return_address(0)) ? CAPMidnightBlue() : color);
}
static void CAPSetFillColor(CAShapeLayer *layer, SEL selector, CGColorRef color) {
    originalFillColor(layer, selector,
        CAPOriginalArrowCaller(__builtin_return_address(0)) ? CAPMidnightBlue() : color);
}

%ctor {
    @autoreleasepool {
        %init;
        Class modelClass = objc_getClass("MLModel");
        if (modelClass) MSHookMessageEx(modelClass, @selector(predictionFromFeatures:error:),
                                        (IMP)CAPModifiedPredict, (IMP *)&originalPredict);
        void *maia = dlsym(RTLD_DEFAULT, "MaiaGo");
        if (maia) MSHookFunction(maia, (void *)CAPMaiaGo, (void **)&originalMaiaGo);
        void *engine = dlsym(RTLD_DEFAULT, "EngineGo");
        if (engine) MSHookFunction(engine, (void *)CAPEngineGo, (void **)&originalEngineGo);
        Class swiftBoard = NSClassFromString(@"_TtC15SwiftChessBoard10ChessBoard");
        if (swiftBoard) {
            Method layout = class_getInstanceMethod(swiftBoard, @selector(layoutSubviews));
            originalSwiftBoardLayout = layout ? (CAPLayoutIMP)method_getImplementation(layout) : nullptr;
            if (originalSwiftBoardLayout)
                MSHookMessageEx(swiftBoard, @selector(layoutSubviews),
                                (IMP)CAPSwiftBoardLayout, (IMP *)&originalSwiftBoardLayout);
        }
        Class webSocketTask = objc_getClass("NSURLSessionWebSocketTask");
        Method send = webSocketTask ?
            class_getInstanceMethod(webSocketTask, @selector(sendMessage:completionHandler:)) : nullptr;
        if (send) {
            originalWebSocketSend = (CAPWebSocketSendIMP)method_getImplementation(send);
            MSHookMessageEx(webSocketTask, @selector(sendMessage:completionHandler:),
                            (IMP)CAPWebSocketSend, (IMP *)&originalWebSocketSend);
        }
        Class shape = objc_getClass("CAShapeLayer");
        if (shape) {
            MSHookMessageEx(shape, @selector(setStrokeColor:), (IMP)CAPSetStrokeColor,
                            (IMP *)&originalStrokeColor);
            MSHookMessageEx(shape, @selector(setFillColor:), (IMP)CAPSetFillColor,
                            (IMP *)&originalFillColor);
        }
        CAPInstallTouchObservers();
        CAPDiscLog(@"diagnostics installed SwiftBoard=%p WebSocket=%p", swiftBoard, webSocketTask);
        NSLog(@"[ChessAssistantPro] Maia=%p Engine=%p", maia, engine);
    }
}
