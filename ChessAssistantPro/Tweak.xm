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
#import <mach/vm_prot.h>
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
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <atomic>
#include <sstream>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

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
static const int kEloPresets[] = {
    400, 600, 800, 900, 1000, 1200, 1320,
    1400, 1600, 1800, 2000, 2400, 2800, 3200
};
static const int kEloPresetCount = sizeof(kEloPresets) / sizeof(kEloPresets[0]);
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

static int CAPEloPresetIndexFor(int value) {
    int best = 0;
    for (int i = 1; i < kEloPresetCount; ++i)
        if (abs(kEloPresets[i] - value) < abs(kEloPresets[best] - value)) best = i;
    return best;
}

static int CAPRequestedStockfishElo(void) {
    id saved = [[NSUserDefaults standardUserDefaults] objectForKey:kStrengthEloKey];
    // Normalize old saved values to the same stops shown by the original panel.
    NSInteger raw = saved && [saved respondsToSelector:@selector(integerValue)] ?
        [saved integerValue] : 3200;
    int value = (int)MAX(400, MIN(3200, raw));
    return kEloPresets[CAPEloPresetIndexFor(value)];
}

// Low-ELO move selection and FEN feature analysis.
// Embedded from Humaneengine.mm (new).txt; move policy and all equation groups.
// Some supplied equation identities are callable diagnostics rather than live
// move pickers. Theos treats unused static diagnostics as build errors.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#define HE_BAND_LO            400
#define HE_BAND_HI            1320         // exclusive
#define HE_MAX_PV             43
#define HE_MAX_TRACKED_DEPTH  64
#define HE_MATE_CP            30000
#define HE_CMD_LEN            96


// Move classes by WIN% lost vs Stockfish's best line (same idea as Game Review)
//    Best/Excellent < 2%      Good 2–5%      Inaccuracy 5–10%       Mistake 10–20%     Blunder 20%+
enum { HE_TOP = 0, HE_GOOD, HE_INACC, HE_MISTAKE, HE_BLUNDER, HE_NCLASS };
static const double HEClassLo[HE_NCLASS] = { 0.0, 2.0,             5.0, 10.0, 20.0 };



// Weaker human-band class targets. Preserve the 600 and 1319 anchors; 400 is
// a continuous extrapolation of the same formula, not a separate move policy.
// These are weights before position-specific availability and human biases.
//                                             top      good   inacc mistake blunder
static const double HEMix600 [HE_NCLASS] = { 0.08, 0.12, 0.20, 0.25, 0.35 };
static const double HEMix1319[HE_NCLASS] = { 0.22, 0.23, 0.23, 0.18, 0.14 };


// Win% for a centipawn score (Lichess curve, close to Game Review)
static double HEWinPct(int cp) {
     if (cp >    2000) cp =   2000;
     if (cp < -2000) cp = -2000;
     return 50.0 + 50.0 * (2.0 / (1.0 + exp(-0.00368208 * cp)) - 1.0);
}


// ===========================================================================
// MARK: - Band parameters (400 → 1319, anchored at 600)
// ===========================================================================


typedef struct {
     double t;                     // -200/719 at 400, 0 at 600, 1 at 1319
     int    depth;                 // search depth                         5 → 9
     int    multipv;               // candidate lines                     43 → 30
     double mix[HE_NCLASS];        // per-move class probabilities
     double spotObvious;           // takes a free piece               0.60 → 0.85
     double spotTactic;            // finds a non-capture win/mate 0.15 → 0.60
    double spotMate1;          // finds mate in 1              0.85 → 1.00
    double visW;               // likes "hangs a piece" blunders 1.5 → 0.5
    int       convThresh;      // winning → fewer errors        500 → 300
    double quietW;             // weight on quiet moves        0.55 → 0.95
    double greedBase, aggroBase, fearBase;
    // H.8: lapse rate — fraction of moves that lapse toward human prior
    double lapseRate;              // 0.03 → 0.01
    // H.9: stochastic depth — geometric parameter for depth mixing
    double depthRho;               // 0.20 → 0.55
    // H.13: within-session sensitivity drift
    double zetaFatigue;            // fatigue coefficient: 0.12 → 0.04
    double zetaError;              // error reaction: 0.08 → 0.03
    double errorThreshold;         // loss threshold triggering reaction
    // H.19: risk attitude (mean-variance)
    double riskAversion;           // 0.05 → 0.20 (always slightly risk-averse)
    // H.20: style tilt magnitudes
    double gammaForce;             // aggressive: captures/checks tilt
    double gammaRisk;              // risk-seeking tilt
    double gammaSimp;              // simplification tilt
    // H.21: strength by position type
    double etaTactical;            // sensitivity shift in tactical positions
    double etaEndgame;             // sensitivity shift in endgames
    // H.24: clock management
    double clockM0;                // base moves remaining
    double clockM1;                // moves-remaining slope
    double clockMmin;              // minimum moves remaining
} HEParams;


static inline int HEHumanBand(int elo) {
    return elo >= HE_BAND_LO && elo < HE_BAND_HI;
}
static inline bool HEPdfBand(int elo) {
    return elo == 1320 || elo == 1400 || elo == 1600 || elo == 1800;
}


static HEParams HEParamsFor(int elo) {
    HEParams p;
    double t = (elo - 600.0) / 719.0;
    if (t < -200.0 / 719.0) t = -200.0 / 719.0;
    if (t > 1.0) t = 1.0;
    p.t = t;


    p.depth          = (int)lround(6.0 + 3.0 * t);
    p.multipv        = (int)lround(40.0 - 10.0 * t);
    for (int c = 0; c < HE_NCLASS; c++)
          p.mix[c]   = HEMix600[c] + (HEMix1319[c] - HEMix600[c]) * t;
    p.spotObvious = 0.60 + 0.25 * t;
    p.spotTactic     = 0.15 + 0.45 * t;
    p.spotMate1      = 0.85 + 0.15 * t;
    p.visW           = 1.50 - 1.00 * t;
    p.convThresh     = (int)lround(500.0 - 200.0 * t);
    p.quietW         = 0.55 + 0.40 * t;
    p.greedBase      = 1.50 - 0.40 * t;
    p.aggroBase      = 1.45 - 0.35 * t;
    p.fearBase       = 0.45 + 0.40 * t;
    // H.8: lapse rate — capped at 0.03 to prevent random queen-hanging moves
    p.lapseRate      = 0.03 - 0.02 * t;
    // H.9: stochastic depth — higher rho = deeper average search at higher rating
    p.depthRho       = 0.20 + 0.35 * t;
    // H.13: within-session drift — fatigue and error reaction weaken with rating
    p.zetaFatigue    = 0.12 - 0.08 * t;
    p.zetaError      = 0.08 - 0.05 * t;
    p.errorThreshold = 8.0  - 3.0  * t;
    // H.19: risk attitude — both ends slightly risk-averse (positive),
    // more so at higher rating.  Position-dependent override in HERiskTilt
    // pushes rhoR higher when actually ahead on the board.
    p.riskAversion   = 0.05 + 0.15 * t;
    // H.20: style tilt magnitudes scale with rating
    p.gammaForce     = 0.25 - 0.10 * t;
    p.gammaRisk      = 0.15 - 0.10 * t;
    p.gammaSimp      = -0.05 + 0.10 * t;
    // H.21: strength by position type (negative = stronger than rating suggests)
    p.etaTactical    = 0.15 - 0.10 * t;
    p.etaEndgame     = 0.10 - 0.05 * t;
    // H.24: clock management
    p.clockM0        = 40.0;
    p.clockM1        = 0.30 + 0.10 * t;
    p.clockMmin      = 5.0 + 3.0 * t;
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
    char sq[64];             // '.' = empty, uppercase = white, lowercase = black
    int    whiteToMove;
    int    epSquare;         // -1 if none
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
          default:     return 0;
    }
}


static int HEParseSquare(const char *s) {
    if (!s || !s[0] || !s[1]) return -1;
    int f = s[0] - 'a';
    int r = s[1] - '1';
    if (!HEOnBoard(f, r)) return -1;
    return HESq(f, r);
}


static int HEParseFEN(const char *fen, HEBoard *out) {
    if (!fen || !out) return 0;
    memset(out->sq, '.', sizeof(out->sq));
    out->whiteToMove = 1;
    out->epSquare         = -1;


    int rank = 7, file = 0;
    const char *p = fen;


    for (; *p && *p != ' '; p++) {
          if (*p == '/') { rank--; file = 0; if (rank < 0) return 0; continue; }
          if (isdigit((unsigned char)*p)) { file += *p - '0'; continue; }
          if (file < 0 || file > 7 || rank < 0) return 0;
          out->sq[HESq(file, rank)] = *p;
        file++;
    }


    while (*p == ' ') p++;
    if (*p) { out->whiteToMove = (*p == 'w'); p++; }


    while (*p == ' ') p++;
    while (*p && *p != ' ') p++;           // castling rights
    while (*p == ' ') p++;
    if (*p && *p != '-') out->epSquare = HEParseSquare(p);


    return 1;
}


// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------


static const int HEKnightD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
static const int HEKingD[8][2]      = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};
static const int HEBishopD[4][2] = {{1,1},{1,-1},{-1,-1},{-1,1}};
static const int HERookD[4][2]      = {{0,1},{1,0},{0,-1},{-1,0}};


// Cheapest piece of side `byWhite` attacking sq (0 if none).
static int HECheapestAttacker(const HEBoard *b, int sq, int byWhite) {
    int best = 0;
    int f = HEFile(sq), r = HERank(sq);
#define HE_TAKE(v) do { if (best == 0 || (v) < best) best = (v); } while (0)


    int pr = byWhite ? r - 1 : r + 1;
    char P = byWhite ? 'P' : 'p';
    if (HEOnBoard(f - 1, pr) && b->sq[HESq(f - 1, pr)] == P) return 100;
    if (HEOnBoard(f + 1, pr) && b->sq[HESq(f + 1, pr)] == P) return 100;


    char N = byWhite ? 'N' : 'n';
    for (int i = 0; i < 8; i++) {
        int nf = f + HEKnightD[i][0], nr = r + HEKnightD[i][1];
        if (HEOnBoard(nf, nr) && b->sq[HESq(nf, nr)] == N) { HE_TAKE(320); break; }
    }


    for (int i = 0; i < 4; i++) {
        int nf = f, nr = r;
        for (;;) {
            nf += HEBishopD[i][0]; nr += HEBishopD[i][1];
            if (!HEOnBoard(nf, nr)) break;
            char c = b->sq[HESq(nf, nr)];
            if (c == '.') continue;
            if (c == (byWhite ? 'B' : 'b')) HE_TAKE(330);
            else if (c == (byWhite ? 'Q' : 'q')) HE_TAKE(900);
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
            if (c == (byWhite ? 'R' : 'r')) HE_TAKE(500);
            else if (c == (byWhite ? 'Q' : 'q')) HE_TAKE(900);
            break;
        }
    }


    char K = byWhite ? 'K' : 'k';
    for (int i = 0; i < 8; i++) {
        int nf = f + HEKingD[i][0], nr = r + HEKingD[i][1];
        if (HEOnBoard(nf, nr) && b->sq[HESq(nf, nr)] == K) { HE_TAKE(10000); break; }
    }
#undef HE_TAKE
    return best;
}


static inline int HEAttackedBy(const HEBoard *b, int sq, int byWhite) {
    return HECheapestAttacker(b, sq, byWhite) != 0;
}


static int HEFindKing(const HEBoard *b, int white) {
    char k = white ? 'K' : 'k';
    for (int i = 0; i < 64; i++) if (b->sq[i] == k) return i;
    return -1;
}


// ---------------------------------------------------------------------------
// Move application
// ---------------------------------------------------------------------------


typedef struct { int from, to; char promo; } HEMove;


static HEMove HEParseMove(const char *uci) {
    HEMove m = { -1, -1, 0 };
    if (!uci || strlen(uci) < 4) return m;
    m.from     = HEParseSquare(uci);
    m.to       = HEParseSquare(uci + 2);
    m.promo = (strlen(uci) >= 5) ? (char)tolower((unsigned char)uci[4]) : 0;
    return m;
}


static HEBoard HEApply(const HEBoard *in, HEMove m) {
    HEBoard b = *in;
    if (m.from < 0 || m.to < 0) return b;


    char piece = b.sq[m.from];
    char lower = (char)tolower((unsigned char)piece);
    int     white = HEIsWhitePiece(piece);


    if (lower == 'p' && m.to == in->epSquare && b.sq[m.to] == '.') {
           int capSq = white ? m.to - 8 : m.to + 8;
           if (capSq >= 0 && capSq < 64) b.sq[capSq] = '.';
    }


    if (lower == 'k' && abs(HEFile(m.to) - HEFile(m.from)) == 2) {
        int r=HERank(m.from);
        if (HEFile(m.to)==6) {
            b.sq[HESq(5,r)] = b.sq[HESq(7,r)];
            b.sq[HESq(7,r)] = '.';
        } else if (HEFile(m.to)==2) {
            b.sq[HESq(3,r)] = b.sq[HESq(0,r)];
            b.sq[HESq(0,r)] = '.';
        }
    }
    b.sq[m.to] = m.promo ? (white ? (char)toupper((unsigned char)m.promo) : m.promo) : piece;
    b.sq[m.from] = '.';

    b.epSquare = -1;
    if (lower == 'p' && abs(HERank(m.to) - HERank(m.from)) == 2)
           b.epSquare = white ? m.from + 8 : m.from - 8;


    b.whiteToMove = !in->whiteToMove;
    return b;
}


// ===========================================================================
// MARK: - Move features
// ===========================================================================


typedef struct {
    int isCapture, capturedValue, isRecapture, capturesHigher, givesCheck;
    int attacksQueen, isBackward, isQuiet, isSamePieceAgain, isPawnPush;
    int hangsPiece, isDeveloping, isCastle, isEarlyQueenSortie, isRookPawn;
    int isPromotion, visibleMistake;
} HEFeatures;


static int HELastOwnTo = -1;      // destination of our previous move
static int HEOppLastTo = -1;      // destination of the opponent's last move


static HEFeatures HEExtract(const HEBoard *b, HEMove m) {
    HEFeatures f;
    memset(&f, 0, sizeof(f));
    if (m.from < 0 || m.to < 0) return f;


    char piece     = b->sq[m.from];
    char lower     = (char)tolower((unsigned char)piece);
    int    white   = HEIsWhitePiece(piece);
    char target = b->sq[m.to];


    f.isCapture         = (target != '.') || (lower == 'p' && m.to == b->epSquare);
    f.capturedValue     = (target != '.') ? HEPieceValue(target) : (f.isCapture ? 100 : 0);
    f.isRecapture       = f.isCapture && (m.to == HEOppLastTo);
    f.capturesHigher = f.isCapture && f.capturedValue > HEPieceValue(piece);
    f.isPromotion       = (m.promo != 0);
    f.isPawnPush        = (lower == 'p') && !f.isCapture;
    f.isRookPawn        = (lower == 'p') && (HEFile(m.from) == 0 || HEFile(m.from) == 7);
    f.isCastle          = (lower == 'k') && abs(HEFile(m.to) - HEFile(m.from)) == 2;
    f.isSamePieceAgain = (m.from == HELastOwnTo);


    int dr = HERank(m.to) - HERank(m.from);
    f.isBackward = white ? (dr < 0) : (dr > 0);


    int homeRank = white ? 0 : 7;
    f.isDeveloping          = (lower == 'n' || lower == 'b') && (HERank(m.from) == homeRank);
    f.isEarlyQueenSortie = (lower == 'q') && (HERank(m.from) == homeRank);


    HEBoard after = HEApply(b, m);


    int oppKing     = HEFindKing(&after, !white);
    f.givesCheck = (oppKing >= 0) && HEAttackedBy(&after, oppKing, white);


    char oq = white ? 'q' : 'Q';
    for (int i = 0; i < 64; i++) {
          if (after.sq[i] != oq) continue;
          if (HEAttackedBy(&after, i, white)) f.attacksQueen = 1;
          break;
    }


    f.isQuiet = !f.isCapture && !f.givesCheck && !f.isPromotion && !f.attacksQueen;
    // SEE-lite on the destination square
    int movedValue = HEPieceValue(after.sq[m.to]);
    int cheapest      = HECheapestAttacker(&after, m.to, !white);
    int defended      = HEAttackedBy(&after, m.to, white);
    if (cheapest && (cheapest < movedValue || !defended)) {
        if (f.capturedValue - movedValue < -80) f.hangsPiece = 1;
    }


    // "Visible" = refutation is one move deep: the moved piece hangs, or any
    // other piece is attacked by something cheaper OR is attacked and undefended.
    f.visibleMistake = f.hangsPiece;
    if (!f.visibleMistake) {
        for (int i = 0; i < 64; i++) {
            char c = after.sq[i];
            if (c == '.' || HEIsWhitePiece(c) != white || i == m.to) continue;
            char l = (char)tolower((unsigned char)c);
            if (l == 'p' || l == 'k') continue;
            int ca = HECheapestAttacker(&after, i, !white);
            if (!ca) continue;
            if (ca < HEPieceValue(c) || !HEAttackedBy(&after, i, white)) {
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
        char l = (char)tolower((unsigned char)b->sq[i]);
        if (l == '.' || l == 'p' || l == 'k') continue;
          v += HEPieceValue(l);
    }
    return v;
}


static HEPhase HEPhaseOf(const HEBoard *b, int ply) {
    int npm = HENonPawnMaterial(b);
    if (npm <= 2200) return HEEndgame;
    if (ply < 16 && npm >= 5600) return HEOpening;
    return HEMiddlegame;
}


// How sharp the position is: our attacks, their threats, pressure on their king.
static double HEComplexity(const HEBoard *b, int white) {
    int ours = 0, theirs = 0, nearKing = 0;
    for (int i = 0; i < 64; i++) {
          char c = b->sq[i];
          if (c == '.') continue;
          if (HEIsWhitePiece(c) != white) {
               if (HEAttackedBy(b, i, white)) ours++;
          } else if (tolower((unsigned char)c) != 'k') {
               if (HEAttackedBy(b, i, !white)) theirs++;
          }
    }
    int ok = HEFindKing(b, !white);
    if (ok >= 0) {
          int f = HEFile(ok), r = HERank(ok);
          for (int i = 0; i < 8; i++) {
               int nf = f + HEKingD[i][0], nr = r + HEKingD[i][1];
               if (HEOnBoard(nf, nr) && HEAttackedBy(b, HESq(nf, nr), white)) nearKing++;
          }
    }
    return ours / 6.0 + theirs / 4.0 + nearKing / 4.0;
}


// ===========================================================================
// MARK: - Per-game state
// ===========================================================================


typedef struct {
    double greed, aggro, fear;
    int       count[HE_NCLASS];     // moves played per class
    int       misses;               // opponent blunders not punished
    int       missStreak;           // misses in a row (it notices eventually)
    int       ownMoves;
    int       lastCp, haveLastCp;   // our eval after our previous move
    int       isWhite, init;
    // H.13: within-session drift tracking
    double lastLoss;               // loss of previous move for error reaction
    int    sessionMoves;           // total moves in this session/game
    int    sessionLength;          // expected game length (estimated)
    // H.10: latent attention state (2-state: focused/distracted)
    double attentionFocused;       // probability currently in focused state
    double focusedSigma;           // sensitivity multiplier when focused
    double distractedSigma;        // sensitivity multiplier when distracted
    double transitionToDistracted; // P(distracted | focused)
    double transitionToFocused;    // P(focused | distracted)
} HEState;


static std::recursive_mutex HEMutex;  // protects ALL engine global state, including info lines.
static HEState        HES;
static HEParams HEHP;
static double HEClockSeconds = -1.0; // set with HEClockRemainingSeconds(), optional
static int            HECfgElo = -1;      // last ELO whose options were sent


static void HEStatsReset(void);
static void HECoreNewGame(int playingWhite, int elo) {
    memset(&HES, 0, sizeof(HES));
    HEStatsReset();
    HEHP = HEParamsFor(elo);


    HES.isWhite = playingWhite ? 1 : 0;
    HES.greed         = HEHP.greedBase + HEFrand() * 0.50;
    HES.aggro         = HEHP.aggroBase + HEFrand() * 0.45;
    HES.fear          = HEHP.fearBase     + HEFrand() * 0.30;
    if (playingWhite) HES.aggro *= 1.20; else HES.fear *= 1.15;
    // H.13: session drift
    HES.lastLoss      = 0;
    HES.sessionMoves  = 0;
    HES.sessionLength = 40 + (int)(HEFrand() * 30);
    // H.10: latent attention — start focused
    HES.attentionFocused       = 0.85 + HEFrand() * 0.10;
    HES.focusedSigma           = 0.90 + HEFrand() * 0.10;
    HES.distractedSigma        = 1.30 + HEFrand() * 0.40;
    HES.transitionToDistracted = 0.03 + 0.04 * (1.0 - HEHP.t);
    HES.transitionToFocused    = 0.15 + 0.10 * HEHP.t;


    HEClockSeconds = -1;
    HELastOwnTo = -1;
    HEOppLastTo = -1;
    HECfgElo          = -1;               // resend engine options
    HES.init          = 1;
}


void HENoteOpponentMove(const char *uci) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    HEOppLastTo = HEParseMove(uci).to;
}


// ===========================================================================
// MARK: - UCI plumbing
// ===========================================================================


typedef struct {
    int     cp;              // side-to-move POV, centipawns
    int     mateIn;          // 0 none, >0 we mate, <0 we get mated
    int     depth;
    int     valid;
    char move[8];
} HELine;


static HELine HELines[HE_MAX_PV];
static int        HELineCount = 0;
// Keep each depth's complete MultiPV snapshot. The engine may stop halfway
// through the next depth; the latest line for each index then spans depths.
static HELine HELinesAtDepth[HE_MAX_TRACKED_DEPTH][HE_MAX_PV];
static int HEDepthCounts[HE_MAX_TRACKED_DEPTH];


void HEResetLines(void) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    memset(HELines, 0, sizeof(HELines));
    HELineCount = 0;
    memset(HELinesAtDepth, 0, sizeof(HELinesAtDepth));
    memset(HEDepthCounts, 0, sizeof(HEDepthCounts));
}
// Fills `out` with UCI commands (last one is "go"). Options are only resent
// when the ELO changes or a new game starts.
static int HECoreConfigure(int elo, char out[][HE_CMD_LEN], int max) {
    int n = 0;
#define HE_ADD(...) do { if (n < max) snprintf(out[n++], HE_CMD_LEN, __VA_ARGS__); } while (0)


    HEResetLines();
    int fresh = (elo != HECfgElo);
    HECfgElo = elo;


    if (!HEHumanBand(elo)) {
        if (fresh) {
            HE_ADD("setoption name MultiPV value 1");
            HE_ADD("setoption name Skill Level value 20");
            if (elo < 3190) {
                   HE_ADD("setoption name UCI_LimitStrength value true");
                   HE_ADD("setoption name UCI_Elo value %d", elo);
            } else {
                   HE_ADD("setoption name UCI_LimitStrength value false");
            }
        }
        HE_ADD("go movetime 200");
        return n;
    }


    if (!HES.init) HECoreNewGame(1, elo);
    HEHP = HEParamsFor(elo);


    if (fresh) {
        HE_ADD("setoption name UCI_LimitStrength value false");
        HE_ADD("setoption name Threads value 1");
        HE_ADD("setoption name Skill Level value 20");       // honest scores
        HE_ADD("setoption name MultiPV value %d", HEHP.multipv);
    }
    HE_ADD("go depth %d movetime 1500", HEHP.depth + (int)arc4random_uniform(2));
#undef HE_ADD
    return n;
}


// Parses "info depth D ... multipv K score cp X ... pv MOVE ..."
void HEParseInfo(const char *line) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    if (!line) return;
    if (!strstr(line, " multipv ") || !strstr(line, " pv ")) return;
    if (strstr(line, " lowerbound") || strstr(line, " upperbound")) return;
    int idx = 0, cp = 0, mate = 0, depth = 0, haveScore = 0;
    char mv[8] = {0};


    const char *p = strstr(line, " multipv ");
    idx = atoi(p + 9);
    if (idx < 1 || idx > HE_MAX_PV) return;


    p = strstr(line, " depth ");
    if (p) depth = atoi(p + 7);


    p = strstr(line, " score cp ");
    if (p) { cp = atoi(p + 10); haveScore = 1; }


    p = strstr(line, " score mate ");
    if (p) {
        mate = atoi(p + 12);
        cp = (mate > 0) ? (HE_MATE_CP - mate) : (-HE_MATE_CP - mate);
        haveScore = 1;
    }
    if (!haveScore) return;


    p = strstr(line, " pv ") + 4;
    int n = 0;
    while (*p && *p != ' ' && n < 7) mv[n++] = *p++;
    mv[n] = 0;
    if (n < 4 || n > 5) return;
    HEMove parsed=HEParseMove(mv);
    if(parsed.from<0||parsed.to<0||depth<1) return;
    if(n==5 && mv[4]!='q' && mv[4]!='r' && mv[4]!='b' && mv[4]!='n') return;

    HELine *l = &HELines[idx - 1];
    l->cp        = cp;
    l->mateIn = mate;
    l->depth     = depth;
    l->valid     = 1;
    memcpy(l->move, mv, sizeof(l->move));


    if (idx > HELineCount) HELineCount = idx;
    if (depth < HE_MAX_TRACKED_DEPTH) {
        HELine *snapshot = &HELinesAtDepth[depth][idx - 1];
        if (!snapshot->valid) ++HEDepthCounts[depth];
        *snapshot = *l;
    }
}


// ===========================================================================
// MARK: - Human bias
// ===========================================================================


static double HEHumanBias(const HEFeatures *f, int ply) {
    double t = HEHP.t, inv = 1.0 - t, w = 1.0;


    if (f->isCapture)          w *= HES.greed;
    if (f->isRecapture)        w *= 1.0 + 0.90 * inv;
    if (f->capturesHigher)      w *= 1.0 + 0.60 * inv;
    if (f->givesCheck)          w *= HES.aggro;
    if (f->attacksQueen)        w *= 1.0 + 0.30 * inv;
    if (f->isPromotion)         w *= 1.6;


    if (f->isBackward)          w *= HES.fear;
    if (f->isQuiet)             w *= HEHP.quietW;
    if (f->isSamePieceAgain) w *= 1.0 + 0.35 * inv;
    if (f->isPawnPush)          w *= 1.0 + 0.20 * inv;
    if (f->hangsPiece)          w *= 0.25 + 0.60 * t;


    if (ply < 16) {
        if (f->isDeveloping)          w *= 1.15 + 0.35 * t;
        if (f->isCastle)              w *= 0.65 + 0.45 * t;
        if (f->isEarlyQueenSortie) w *= 1.0 + 0.55 * inv;
        if (f->isRookPawn)            w *= 0.45 + 0.45 * t;
    }


    if (w < 0.01) w = 0.01;
    return w;
}


// Weighted random index over w[0..n), -1 if all zero.
static int HEPick(const double *w, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += w[i];
    if (sum <= 0.0) return -1;
    double r = HEFrand() * sum, acc = 0.0;
    int last = -1;
    for (int i = 0; i < n; i++) {
        if (w[i] <= 0.0) continue;
        last = i;
        acc += w[i];
        if (r < acc) return i;
    }
    return last;
}


// ============================================================================
// PDF MATHEMATICS, equations 01-75 (Chess Decision Mathematics, Nov 2026)
// EQUATION COVERAGE MAP (same source-group numbering as uploaded PDF)
// 01 Value-scaled move loss: phi/losses
// 02 Move probability relation: choice
// 03 Probability normalization: Law::g and choice
// 04 Expected error and rating index: expectedError/intrinsicRating/Tracker
// 05 Score moments: mean/variance
// 06 Mean-reverting running total: revert
// 07 Stationary variance identities: stationaryVariance/incrementVariance
// 08 Adjacent-increment dependence: lagOneCovariance/lagOneCorrelation/Tracker
// 09 Log-time distribution: timeLogMean/logNormalLogDensity
// 10 Sensitivity limits: sensitivityLimits
// 11 Rating-error inversion: targetError
// 12 Depth-constrained choice rule: depthPolicy
// 13 First-passage decision process: Diffusion/simulateFirstExit
// 14 First-exit time density: exitDensity
// 15 Expected decision duration: expectedExit
// 16 Position-dependent drift and boundary: diffusion
// 17 Exponential choice family: tilt/logPartition
// 18 Cumulant moments: mean/variance
// 19 Relative-entropy expansions: kl/logPartition
// 20 Local log-likelihood expansion: localLogLR
// 21 Standardized performance score: standardizedScore
// 22 Rating-to-score conversion: eloToScoreGain
// 23 Joint move-time distributions: jointLog
// 24 Conditional information chain rule: jointKL
// 25 Additive divergence identity: jointKL
// 26 Per-observation log ratio: logOneStepRatio
// 27 Log-normal information distance: logNormalKL
// 28 Sequential ratio product: appendLogEvidence
// 29 Uniform crossing bound: crossed
// 30 Expected log-growth: kl/Tracker
// 31 Approximate threshold time: thresholdObservations
// 32 Admissible rating path: ratingStepAllowed/maxLogPath
// 33 Path-conditional likelihood: pathConditionalLog
// 34 Composite-normalized ratio: compositeLogEvidence
// 35 Decomposition by rating variation: compositeDecomposition
// 36 Two-sided information bounds: minimumSamples
// 37 Binary relative entropy: binaryKL
// 38 Minimum observation bound: minimumSamples
// 39 Effective model divergence: minFamilyKL
// 40 Rating-adjusted sequential ratio: compositeLogEvidence
// 41 Rating-aware choice perturbation: tilt/ratingScore
// 42 Choice and rating scores: scoreTilt/ratingScore
// 43 Parameter derivatives: scaleScore/ratingScore
// 44 Closed-form rating score: scaleScore
// 45 Fisher information matrix: fisher
// 46 Local effective information: Fisher::effective
// 47 Information collapse condition: informationCollapse
// 48 Orthogonal perturbation component: orthogonal/effectivePerturbation
// 49 Boundary-constrained information: constrainedInformation
// 50 Joint-channel information gain: informationWithTiming
// 51 One-step model mismatch: chiSquared/mismatchCovarianceBound
// 52 Expected log-ratio drift: expectedLogDrift
// 53 Fractional-power crossing bound: fractionalCrossing
// 54 Tilted moment identity: tiltedMoment
// 55 Calibration inflation exponent: inflationExponent
// 56 Critical model distance: criticalMismatch
// 57 History-averaged conditional law: historyAverage
// 58 Conditional-information decomposition: conditionalMutualInformation
// 59 History-conditional likelihood expectation: historyConditionalRatioMean
// 60 History-aware ratio product: historySequentialLog
// 61 Regime-to-regime divergence: clockRegimeKL
// 62 Change in choice information: clockRegimeKL/kl
// 63 Sensitivity information score: scaleScore/sensitivityFisher
// 64 Inverse-Gaussian divergence: inverseGaussianKL
// 65 Transition-state subdensity: transitionStateDensity
// 66 Post-transition exit-time law: postTransitionExit
// 67 Piecewise boundary-crossing time: changedBoundaryTime
// 68 Clock-conditioned observation law: clockConditionedJoint
// 69 Complete conditional factorization: jointLog/clockConditionedJoint
// 70 Generalized path statistic: compositeLogEvidence/generalizedThreshold
// 71 Model-class sample bound: effectiveSampleLower
// 72 Second-order sample bound: secondOrderSampleBound
// 73 Normalization derivatives: Law::g/gp/gpp
// 74 Newton iteration: choice (safeguarded Newton)
// 75 Quadratic convergence guarantee: Law::newtonQuadraticBound
// NOTE: analytic bounds, hypothesis tests and alternative timing laws are
// callable mathematical diagnostics; they are NOT additional move pickers.
// H0 coefficients and empirical human rating data were not supplied.
// Sequential Testing of Choice and Response-Time Models: same derivations.
// All 75 numbered equation groups are represented by executable operations,
// bounds or diagnostic identities. Only genuine choice and timing distributions
// influence the live policy: the other equations measure/validate it. They NEVER manufacture
// human calibration, clock readings, population data, or a verified H0 model.
// All density arguments use seconds, evaluations use PAWNS, probabilities [0,1].
//
// SECTION 1 EQUATIONS APPLIED TO 400-1319 BAND (HECoreChooseMove):
// H.1  Gibbs structure     — implicit in choice(); β=ln(1/p0), Z(β)=1
// H.2  Comparative statics — sensitivity ladder via fitLaw
// H.3  Random utility      — logit equivalence; σ=1/β
// H.4  Information cost    — tilt() with θ=1/λ
// H.5  Performance stats   — Tracker::record
// H.6  Intrinsic rating    — observedIPR/expectedIPR
// H.7  Fisher geometry     — fisher(), scaleScore()
// H.8  Lapse model         — HEApplyLapse: human-prior gated, blocks >15pp blunders LIVE
// H.9  Stochastic depth    — HEDepthWeights (geometric mixture) LIVE
// H.10 Latent attention    — HEAttentionUpdate (2-state Markov) LIVE
// H.11 Speed-accuracy      — Hick-Hyman via drift-diffusion threshold
// H.12 LBA                 — available as diagnostic
// H.13 Within-session drift— HESensitivityDrift (fatigue+error) LIVE
// H.14 Elo dynamics        — eloToScoreGain diagnostic
// H.15 Glicko              — diagnostic
// H.16 Win expectancy      — HEWinPct logistic map LIVE
// H.17 Calibration test    — Tracker calibration diagnostics
// H.18 Rating ladder       — KL divergence between ratings
// H.19 Risk attitude       — HERiskTilt: risk-averse when ahead + simplify bonus LIVE
// H.20 Playing styles      — HEStyleTilt: feature tilt + engine-quiet penalty LIVE
// H.21 Position type       — HEClassifyPosition + etaShift LIVE
// H.22 Player types        — available as diagnostic
// H.23 Opponent modelling  — available as diagnostic
// H.24 Clock management    — HEMoveBudget (per-move budget) LIVE
// H.25 Neural models       — Maia ONNX path (separate)
// H.26 Model detection     — KL rate diagnostic
// H.27 Complete law        — composite s with η, ζ, σ_k assembled LIVE
// ============================================================================
namespace HEMath {
using Vec = std::vector<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tiny = 1.0e-300;
static double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
static double safeLog(double x) { return std::log(std::max(x, tiny)); }
static double sq(double x) { return x*x; }
static double total(const Vec& a) { return std::accumulate(a.begin(),a.end(),0.0); }
static Vec normalize(Vec a) {
    double sum=0; for (double &x:a) {x=std::isfinite(x)?std::max(0.0,x):0.0;sum+=x;}
    if(a.empty())return a;
    if(sum<=0){std::fill(a.begin(),a.end(),1.0/a.size());return a;}
    for(double &x:a)x/=sum;return a;
}
static double dot(const Vec &a,const Vec &b) {
    if(a.size()!=b.size()){NSLog(@"[ChessAssistantPro] HEMath dot: vector size mismatch");return 0;}
    double v=0;for(size_t i=0;i<a.size();++i)v+=a[i]*b[i];return v;
}
static double entropy(const Vec &p) {double h=0;for(double v:p)if(v>0)h-=v*std::log(v);return h;}
// 01: scaled score loss and sign-safe transform.
static double phi(double v){return std::copysign(std::log1p(std::fabs(v)),v);}
static Vec losses(const Vec& v) {Vec d(v.size());if(v.empty())return d;
    double top=*std::max_element(v.begin(),v.end());
    for(size_t i=0;i<v.size();++i)d[i]=std::max(0.0,phi(top)-phi(v[i]));return d;
}
// 02,03,73-75: implicit Regan-Haworth normalization with analytic g', g'';
// safeguarded Newton + bisection; equality, extremes, and overflow covered.
struct Law {Vec p,w,d;double p0=1,s=0,c=0;int iterations=0;
    double g(double x)const{double z=0;for(double wi:w)z+=std::exp(wi*safeLog(x));return z;}
    double gp(double x)const{double z=0;for(double wi:w)z+=wi*std::exp((wi-1)*safeLog(x));return z;}
    double gpp(double x)const{double z=0;for(double wi:w)z+=wi*(wi-1)*std::exp((wi-2)*safeLog(x));return z;}
    double newtonQuadraticBound(double rootError,double high)const{
        // Eq 75: M=max_{p in [p*,high]} g''(p), not merely g''(high).
        // For 1<w<2 the curvature DECREASES; for w>2 it increases.
        if(!(high>=p0&&high<=1.0&&p0>0))return NAN;
        double M=0.0;
        for(double wi:w)if(wi>1.0){
            const double at=(wi<2.0)?p0:high;
            M+=wi*(wi-1.0)*std::exp((wi-2.0)*safeLog(at));
        }
        return M/std::max(2*gp(p0),tiny)*sq(rootError);
    }
};
static Law choice(const Vec &d,double s,double c){
    if(!(s>0&&c>0)||!std::isfinite(s)||!std::isfinite(c)){NSLog(@"[ChessAssistantPro] HEMath choice: positive finite s,c required");return out;}
    Law out;out.d=d;out.s=s;out.c=c;out.w.resize(d.size());out.p.resize(d.size());
    if(d.empty())return out;
    for(size_t i=0;i<d.size();++i){
        if(d[i]<0||!std::isfinite(d[i])){NSLog(@"[ChessAssistantPro] HEMath choice: invalid loss at %zu",i);return out;}
        double exponent=std::pow(d[i]/s,c);
        out.w[i]=std::exp(std::min(exponent,40.0)); // numerical cap; not statistical calibration
    }
    double low=0.0,high=1.0,x=1.0/out.w.size();
    for(int it=0;it<160;++it){
        double gx=out.g(x)-1.0;out.iterations=it+1;
        if(std::fabs(gx)<1e-13||high-low<1e-15)break;
        if(gx>0)high=x;else low=x;
        double dfx=out.gp(x);double y=(dfx>tiny)?x-gx/dfx:std::numeric_limits<double>::quiet_NaN();
        x=(std::isfinite(y)&&y>low&&y<high)?y:0.5*(low+high);
    }
    out.p0=x;
    for(size_t i=0;i<d.size();++i)out.p[i]=std::exp(out.w[i]*safeLog(x));
    out.p=normalize(out.p);
    return out;
}
// Section 8.3: the PDF's independent five-move example is a regression check
// for the root and probability ordering. It is not a low-Elo calibration.
static bool referenceSolverCheck(){
    const Law law=choice({0.0,.04,.12,.28,.55},.098,.483);
    const double expected[]={.5264203468554158,.2930243324562274,
        .1447172973036302,.0341823572108512,.0016556661738753};
    if(law.p.size()!=5||law.iterations>20||
       std::fabs(law.g(law.p0)-1.0)>1.e-12)return false;
    for(size_t i=0;i<5;++i)
        if(std::fabs(law.p[i]-expected[i])>1.e-10)return false;
    return true;
}
// 04,11: historical intrinsic performance rating and inversion (not actual Elo).
static double expectedError(const Vec&p,const Vec&d){return dot(p,d);}
static double intrinsicRating(double ae){return 3571.0-15413.0*ae;}
static double targetError(int rating){return std::max(0.0,(3571.0-rating)/15413.0);}
// 05,18: conditional moments and Fisher variance in outcome quality.
static double mean(const Vec&p,const Vec&u){return dot(p,u);}
static double variance(const Vec&p,const Vec&u){double m=mean(p,u),v=0;
    for(size_t i=0;i<p.size();++i)v+=p[i]*sq(u[i]-m);return std::max(0.0,v);}
// 06-08: stationary mean-reverting model / lag-1 regularity diagnostic.
static double revert(double M,double e,double k){return (1-k)*M+e;}
static double stationaryVariance(double k){return 1.0/(k*(2-k));}
static double incrementVariance(double k){return 2.0/(2.0-k);}
static double lagOneCovariance(double k){return -k/(2.0-k);}
static double lagOneCorrelation(double k){return -0.5*k;}
// 09: log-normal decision-time law, conditional on entropy + remaining clock.
static double timeLogMean(double b0,double b1,double b2,double H,double clockSec){
    return b0+b1*H+b2*std::log(std::max(clockSec,0.001));}
static double normalLogDensity(double y,double m,double sd){
    if(!(sd>0))return -INFINITY;return -.5*sq((y-m)/sd)-std::log(sd)-.5*std::log(2*pi);}
static double logNormalLogDensity(double elapsed,double motor,double m,double sd){
    if(elapsed<=motor)return -INFINITY;
    double t=elapsed-motor;return normalLogDensity(std::log(t),m,sd)-std::log(t);
}
// 10: limits are tested by choice(d,s,c), s->0 and s->infinity.
// 12: deterministic depth-limited point-mass policy for comparison, not generation.
static Vec depthPolicy(const Vec&v){Vec q(v.size(),0);if(!v.empty())q[std::max_element(v.begin(),v.end())-v.begin()]=1;return q;}
// 10: compute both limits explicitly using the exact same normalization solver.
static std::pair<Vec,Vec> sensitivityLimits(const Vec&d,double consistency){
    return {choice(d,1.e-8,consistency).p,choice(d,1.e8,consistency).p};
}
// 13-16: drift diffusion, first-exit density/mean, entropy/clock-dependent drift.
struct Diffusion {double mu=0,B=1,sigma=1,motor=0;};
static double expectedExit(const Diffusion &a){if(a.B<=0||a.sigma<=0)return NAN;
    if(std::fabs(a.mu)<1e-8)return sq(a.B/a.sigma);
    return a.B/a.mu*std::tanh(a.mu*a.B/sq(a.sigma));}
static double exitDensity(double t,const Diffusion&a,int terms=160){
    if(!(t>0&&a.B>0&&a.sigma>0))return 0;
    double z=0,kappa=sq(pi*a.sigma)/(8*sq(a.B));
    for(int k=0;k<terms;++k){double term=(2*k+1)*std::exp(-sq(2*k+1)*kappa*t);
        z+=(k&1)?-term:term;if(k>10&&term<1e-14)break;}
    double h=a.mu*a.B/sq(a.sigma), pref;
    // log(cosh) avoids overflow.
    double logcosh=std::fabs(h)+std::log1p(std::exp(-2*std::fabs(h)))-std::log(2.0);
    pref=std::exp(logcosh-sq(a.mu)*t/(2*sq(a.sigma)));
    return std::max(0.0,pref*kappa*4/pi*z);
}
static Diffusion diffusion(double entropyX,double clock,double secondGap,double sensitivity,
                            double b0,double b1,double b2,double m0,double m1){
    Diffusion a; a.B=std::max(.05,b0+b1*entropyX+b2*std::log(std::max(clock,.001)));
    a.mu=m0+m1*secondGap/std::max(sensitivity,1e-10);return a;}
// 13: direct first passage of a drifted Wiener process; optional simulation
// for testing the theoretical density against Monte Carlo, not move generation.
static double simulateFirstExit(const Diffusion&a,std::mt19937_64 &rng,double dt=.002){
    if(!(dt>0&&a.B>0&&a.sigma>0))return NAN;
    std::normal_distribution<double> normal(0,1);
    double x=0;
    for(int k=1;k<=5000000;++k){
        x+=a.mu*dt+a.sigma*std::sqrt(dt)*normal(rng);
        if(std::fabs(x)>=a.B)return k*dt;
    }
    return NAN; // right-censored simulation, caller must account for censoring
}
// 17-20: exponential tilt, cumulants, KL and local score expansion.
static Vec tilt(const Vec&p,const Vec&q,double theta){
    if(p.size()!=q.size()){NSLog(@"[ChessAssistantPro] HEMath tilt: vector size mismatch");return normalize(Vec(p.size(),1));}
    Vec logs(p.size()),r(p.size());double mx=-INFINITY;
    for(size_t i=0;i<p.size();++i){logs[i]=safeLog(p[i])+theta*q[i];mx=std::max(mx,logs[i]);}
    for(size_t i=0;i<p.size();++i)r[i]=std::exp(logs[i]-mx);return normalize(r);
}
static double logPartition(const Vec&p,const Vec&q,double theta){
    double mx=-INFINITY;for(size_t i=0;i<p.size();++i)mx=std::max(mx,safeLog(p[i])+theta*q[i]);
    double sum=0;for(size_t i=0;i<p.size();++i)sum+=std::exp(safeLog(p[i])+theta*q[i]-mx);
    return mx+std::log(std::max(sum,tiny));
}
static double kl(const Vec&a,const Vec&b){if(a.size()!=b.size()){NSLog(@"[ChessAssistantPro] HEMath KL size mismatch");return 0;}
    double d=0;for(size_t i=0;i<a.size();++i)if(a[i]>0)d+=a[i]*(std::log(a[i])-safeLog(b[i]));return std::max(0.0,d);}
static double localLogLR(double theta,double qMinusMu,double v){return theta*qMinusMu-.5*sq(theta)*v;}
// 21: standardized score statistic; 22: Elo-to-expected-score gain.
static double standardizedScore(double sumCentered,double sumV){return sumV>0?sumCentered/std::sqrt(sumV):0;}
static double eloToScoreGain(double deltaR){return .5*std::tanh(std::log(10.0)*deltaR/800.0);}
// 23-27: joint choice/time likelihood, information chain and Gaussian time KL.
static double jointLog(double moveProbability,double logTimePdf){return safeLog(moveProbability)+logTimePdf;}
static double jointKL(const Vec&piX,const Vec&p0X,const Vec&conditionalKL){return kl(piX,p0X)+dot(piX,conditionalKL);}
static double logOneStepRatio(double q,double p,double f1Log,double f0Log){
    return safeLog(q)-safeLog(p)+f1Log-f0Log;}
static double logNormalKL(double m1,double m0,double sd){return sq(m1-m0)/(2*sq(sd));}
// 28-31: log-domain sequential product and always-valid crossing threshold,
// conditional on correctly specified p0 and a predictable alternative.
static double appendLogEvidence(double cumulative,double step){return cumulative+step;}
static double logThreshold(double alpha){return -std::log(alpha);}
static bool crossed(double evidence,double alpha){return evidence>=logThreshold(alpha);}
static double thresholdObservations(double alpha,double d){return d>0?logThreshold(alpha)/d:INFINITY;}
// 32-35,39-40,70: composite rating model and bounded rating paths.
static bool ratingStepAllowed(int prev,int next,int lo,int hi,int velocity){
    return next>=lo&&next<=hi&&prev>=lo&&prev<=hi&&std::abs(next-prev)<=velocity;}
static double maxLogPath(const std::vector<Vec>&ll,const Vec&ratings,double maxVelocity){
    if(ll.empty())return 0;
    size_t n=ratings.size();if(n==0)return -INFINITY;
    Vec previous=ll.front();if(previous.size()!=n){NSLog(@"[ChessAssistantPro] HEMath rating lattice mismatch");return -INFINITY;}
    for(size_t t=1;t<ll.size();++t){
        if(ll[t].size()!=n){NSLog(@"[ChessAssistantPro] HEMath rating path mismatch");return -INFINITY;}
        Vec next(n,-INFINITY);
        for(size_t j=0;j<n;++j)for(size_t k=0;k<n;++k)
            if(std::fabs(ratings[j]-ratings[k])<=maxVelocity)
                next[j]=std::max(next[j],previous[k]+ll[t][j]);
        previous.swap(next);
    }
    return *std::max_element(previous.begin(),previous.end());
}
static double compositeLogEvidence(double numerator,const std::vector<Vec>&ll,const Vec&ratings,double maxVelocity){
    return numerator-maxLogPath(ll,ratings,maxVelocity);}
static double compositeDecomposition(double lnNumerator,double unrestricted,double constrained){
    return (lnNumerator-unrestricted)+(unrestricted-constrained);}
// 33: one-game path-conditional log-likelihood. Missing clock data is not
// silently estimated: use a move-only conditional law when timeObserved=false.
static double pathConditionalLog(double pMove,double baselineLogTime,bool timeObserved){
    return safeLog(pMove)+(timeObserved?baselineLogTime:0.0);
}
// 39: effective divergence to a finite reference family of rating profiles.
static double minFamilyKL(const Vec&alternative,const std::vector<Vec>&ratingLaws){
    if(ratingLaws.empty())return NAN;
    double d=INFINITY;
    for(const Vec&law:ratingLaws)d=std::min(d,kl(alternative,law));
    return d;
}
// 36-38,71: binary KL and necessary sample sizes for alpha / power.
static double binaryKL(double p,double q){
    if(!(p>=0&&p<=1&&q>=0&&q<=1)){NSLog(@"[ChessAssistantPro] HEMath binaryKL: probabilities outside [0,1]");return 0;}
    if(q==0)return p==0?0:INFINITY;
    if(q==1)return p==1?0:INFINITY;
    p=clamp(p,0,1);q=clamp(q,tiny,1-tiny);
    return (p>0?p*std::log(p/q):0)+(p<1?(1-p)*std::log((1-p)/(1-q)):0);
}
static double minimumSamples(double alpha,double beta,double kl10,double kl01){
    if(!(alpha>0&&alpha<1&&beta>0&&beta<1))return INFINITY;
    if(kl10<=0||kl01<=0)return INFINITY;
    return std::max(binaryKL(1-beta,alpha)/kl10,binaryKL(alpha,1-beta)/kl01);
}
static double effectiveSampleLower(double alpha,double beta,double dEff,double lambda=1){
    if(!(dEff>0&&lambda>0))return INFINITY;
    return binaryKL(1-beta,std::pow(alpha,1/lambda))/dEff;
}
// 41-44,63: rating scores and scale sensitivity (analytic and finite-difference).
static Vec scaleScore(const Law&l){
    Vec r(l.p.size(),0);double mnum=0,mden=0;
    for(size_t i=0;i<r.size();++i){mnum+=l.p[i]*l.w[i]*std::log(l.w[i]);mden+=l.p[i]*l.w[i];}
    double avg=mden>0?mnum/mden:0;
    double a=l.c/l.s*std::log(1/std::max(l.p0,tiny));
    for(size_t i=0;i<r.size();++i)r[i]=a*(l.w[i]*std::log(l.w[i])-l.w[i]*avg);
    return r;
}
static Vec ratingScore(const Law&l,double dsByRating){Vec r=scaleScore(l);for(double &v:r)v*=dsByRating;return r;}
static Vec scoreTilt(const Vec&q,const Vec&p){Vec a(q.size());double m=mean(p,q);
    for(size_t i=0;i<a.size();++i)a[i]=q[i]-m;return a;}
// 45-50: Fisher matrix, orthogonal information, constrained rate corridor.
struct Fisher {double rr=0,rt=0,tt=0;
    double effective()const{return rr>tiny?std::max(0.0,tt-sq(rt)/rr):tt;}
    double correlationSquared()const{return rr*tt>tiny?clamp(sq(rt)/(rr*tt),0,1):0;}
};
static Fisher fisher(const Vec&p,const Vec&sr,const Vec&st){
    Fisher f;for(size_t i=0;i<p.size();++i){f.rr+=p[i]*sq(sr[i]);f.rt+=p[i]*sr[i]*st[i];f.tt+=p[i]*sq(st[i]);}return f;
}
static bool informationCollapse(const Fisher&f,double threshold=1e-6){return f.tt>tiny&&f.effective()<threshold*f.tt;}
static Vec orthogonal(const Vec&h,const Vec&sr,const Vec&p){
    double varR=0,cov=0;
    for(size_t i=0;i<p.size();++i){varR+=p[i]*sq(sr[i]);cov+=p[i]*h[i]*sr[i];}
    Vec out=h;double coef=varR>tiny?cov/varR:0;
    for(size_t i=0;i<p.size();++i)out[i]-=coef*sr[i];return out;
}
static double effectivePerturbation(double epsilon,const Vec&p,const Vec&hPerp){
    Vec square(hPerp.size());for(size_t i=0;i<hPerp.size();++i)square[i]=sq(hPerp[i]);
    return .5*sq(epsilon)*dot(p,square);
}
static double constrainedInformation(double theta,const Fisher&f,double unconstrainedR,double maxR){
    return .5*sq(theta)*f.effective()+.5*f.rr*sq(std::max(0.0,unconstrainedR-maxR));
}
static double informationWithTiming(const Fisher&f,double timingRatingInfo){
    return f.tt-sq(f.rt)/std::max(tiny,f.rr+std::max(0.0,timingRatingInfo));}
// 51-56: misspecification sensitivity and e-process calibration diagnostics.
static double chiSquared(const Vec&p,const Vec&q){double v=0;
    for(size_t i=0;i<p.size();++i)v+=sq(p[i]-q[i])/std::max(q[i],tiny);return v;}
static double mismatchCovarianceBound(double chiAlt,double chiTrue){return std::sqrt(chiAlt*chiTrue);}
static double expectedLogDrift(double klTrueModel,double klTrueAlt){return klTrueModel-klTrueAlt;}
static double fractionalCrossing(double alpha,double lambda){return std::pow(alpha,lambda);}
static double tiltedMoment(const Vec&p,const Vec&q,double theta0,double theta,double lambda){
    return std::exp(logPartition(p,q,theta0+lambda*theta)
        -logPartition(p,q,theta0)-lambda*logPartition(p,q,theta));
}
static double inflationExponent(double mismatch,double altKL){
    return altKL>0?clamp(1-2*std::sqrt(std::max(0.0,mismatch)/altKL),0,1):0;}
static double criticalMismatch(double altKL){return .25*altKL;}
// 57-60: history-conditional projection and conditional-information decomposition.
static Vec historyAverage(const std::vector<Vec>&laws,const Vec&freq){
    if(laws.empty())return {};
    if(laws.size()!=freq.size()){NSLog(@"[ChessAssistantPro] HEMath historyAverage: history/weight mismatch");return {};}
    for(const Vec& law:laws)if(law.size()!=laws.front().size())
        {NSLog(@"[ChessAssistantPro] HEMath historyAverage: law size mismatch");return {};}
    Vec a(laws[0].size(),0);Vec freqN=normalize(freq);
    for(size_t k=0;k<laws.size();++k)for(size_t i=0;i<a.size();++i)a[i]+=freqN[k]*laws[k][i];
    return normalize(a);
}
static double conditionalMutualInformation(const std::vector<Vec>&laws,const Vec&freq){
    Vec avg=historyAverage(laws,freq),f=normalize(freq);double mi=0;
    for(size_t i=0;i<laws.size();++i)mi+=f[i]*kl(laws[i],avg);return mi;}
static double historyConditionalRatioMean(const Vec&historyLaw,const Vec&alternative,const Vec&baseline){
    if(historyLaw.size()!=alternative.size()||alternative.size()!=baseline.size())
        {NSLog(@"[ChessAssistantPro] HEMath historyConditionalRatioMean: law size mismatch");return 0;}
    double out=0;
    for(size_t i=0;i<historyLaw.size();++i)
        out+=historyLaw[i]*alternative[i]/std::max(baseline[i],tiny);
    return out;
}
// 60: history-conditional sequential product; ratios supplied per history
// and state after conditioning, so the environment transition cancels.
static double historySequentialLog(const Vec&historyLogRatios){return total(historyLogRatios);}
// 61-64: two-regime information, sensitivity and inverse-Gaussian KL.
static double clockRegimeKL(const Vec&p2,const Vec&p1,const Vec&timeKL){return jointKL(p2,p1,timeKL);}
static double sensitivityFisher(const Law&l){Vec s=scaleScore(l);return variance(l.p,s);}
// 43: derivatives with respect to rating, assuming ds/dR supplied by fit.
static Vec exponentRatingDerivative(const Law&l,double dsByRating){
    Vec out(l.w.size());
    for(size_t i=0;i<out.size();++i)
        out[i]=-l.c*dsByRating/l.s*l.w[i]*std::log(l.w[i]);
    return out;
}
static Vec logProbabilityRatingDerivative(const Law&l,double dsByRating){
    Vec out(l.p.size());
    if(out.empty())return out;
    const Vec dw=exponentRatingDerivative(l,dsByRating);
    double denom=dot(l.p,l.w);
    double dp0=-safeLog(l.p0)*dot(l.p,dw)/std::max(denom,tiny);
    for(size_t i=0;i<out.size();++i)
        out[i]=dw[i]*safeLog(l.p0)+l.w[i]*dp0;
    return out;
}
// 51: E_true[L]=1+Cov_p0(L,p_true/p0), when L=q/p0.
static double mismatchExpectation(const Vec&trueLaw,const Vec&alternative,const Vec&reference){
    if(trueLaw.size()!=alternative.size()||reference.size()!=alternative.size())
        {NSLog(@"[ChessAssistantPro] HEMath mismatchExpectation: law size mismatch");return 0;}
    double v=0;
    for(size_t i=0;i<trueLaw.size();++i)
        v+=trueLaw[i]*alternative[i]/std::max(reference[i],tiny);
    return v;
}
// 58: E_H[KL(p_h || H0)] = I(A;H | x) + KL(p_bar || H0).
static double historyInformationDecomposition(const std::vector<Vec>&laws,
        const Vec&frequencies,const Vec&reference){
    return conditionalMutualInformation(laws,frequencies)
        +kl(historyAverage(laws,frequencies),reference);
}
// 62: exact regime KL and its second-order local sensitivity approximation.
static std::pair<double,double> sensitivityRegimeDivergence(const Law&start,
        const Law&end){
    if(start.p.size()!=end.p.size()){NSLog(@"[ChessAssistantPro] HEMath sensitivityRegimeDivergence: size mismatch");return {0,0};}
    return {kl(end.p,start.p),.5*sq(end.s-start.s)*sensitivityFisher(start)};
}
static double inverseGaussianKL(double m2,double shape2,double m1,double shape1){
    // Eq 64, IG(mean m2, shape2) || IG(mean m1, shape1).
    if(m1<=0||m2<=0||shape1<=0||shape2<=0)return NAN;
    return .5*(std::log(shape1/shape2)+shape2/shape1-1.0
                +shape1*sq(m2-m1)/(sq(m1)*m2));
}
// 65-67: two-regime absorbed diffusion (eigen expansions + quadrature).
// These are only evaluated if a caller supplies clock-change parameters.
static double transitionStateDensity(double y,double t,const Diffusion&a,int terms=180){
    if(!(t>0&&a.B>0&&std::fabs(y)<a.B))return 0;
    double z=0;for(int n=1;n<=terms;++n){
        double term=std::sin(n*pi/2)*std::sin(n*pi*(y+a.B)/(2*a.B))
            *std::exp(-sq(n*pi*a.sigma)*t/(8*sq(a.B)));
        z+=term;if(n>24&&std::fabs(term)<1e-14)break;
    }
    return std::max(0.0,std::exp(a.mu*y/sq(a.sigma)-sq(a.mu)*t/(2*sq(a.sigma)))*z/a.B);
}
static double postTransitionExit(double t,double y,const Diffusion&a,int terms=180){
    if(!(t>0&&a.B>0&&std::fabs(y)<a.B))return 0;
    double sum=0;
    for(int n=1;n<=terms;++n){
        double a1=-a.mu*(a.B+y)/sq(a.sigma),a2=a.mu*(a.B-y)/sq(a.sigma);
        double term=n*std::sin(n*pi*(y+a.B)/(2*a.B))
            *std::exp(-sq(n*pi*a.sigma)*t/(8*sq(a.B)))
            *(std::exp(clamp(a1,-700,700))+((n&1)?1:-1)*std::exp(clamp(a2,-700,700)));
        sum+=term;
        if(n>24&&std::fabs(term)<1e-13)break;
    }
    return std::max(0.0,std::exp(-sq(a.mu)*t/(2*sq(a.sigma)))*pi*sq(a.sigma)*sum/(4*sq(a.B)));
}
static double integrateMidpoint(const std::function<double(double)>&f,double l,double r,int steps=240){
    if(r<=l)return 0;double d=(r-l)/steps,sum=0;
    for(int i=0;i<steps;++i)sum+=f(l+(i+.5)*d);return sum*d;
}
struct ChangedBoundary {double density=0,pointMass=0;};
static ChangedBoundary changedBoundaryTime(double t,double changeTime,const Diffusion&before,const Diffusion&after){
    if(!(before.B>=after.B&&after.B>0&&changeTime>0))return {};
    if(t<changeTime)return {exitDensity(t,before),0};
    auto g=[&](double y){return transitionStateDensity(y,changeTime,before);};
    double atom=integrateMidpoint(g,-before.B,-after.B)+integrateMidpoint(g,after.B,before.B);
    if(t==changeTime)return {0,atom};
    double continuous=integrateMidpoint([&](double y){return g(y)*postTransitionExit(t-changeTime,y,after);},-after.B,after.B);
    return {continuous,atom};
}
// 68-70: clock-conditioned complete conditional law and generalized statistic.
static double clockConditionedJoint(double moveP,double logTime){return std::exp(jointLog(moveP,logTime));}
static double generalizedThreshold(double alpha,double lambda){
    if(!(alpha>0&&alpha<1&&lambda>0&&lambda<=1))return INFINITY;
    return std::log(1/alpha)/lambda;
}
// 71-72: model class and second-order lower sample bound.
static double secondOrderSampleBound(double alpha,double beta,double lambda,double theta,const Fisher&f,
                                      double boundaryPenalty){
    double d=.5*sq(theta)*f.effective()+boundaryPenalty;
    if(!(lambda>0&&d>0))return INFINITY;
    double H=-(beta>0?beta*std::log(beta):0)-((1-beta)>0?(1-beta)*std::log(1-beta):0);
    return std::max(0.0,((1-beta)*std::log(1/alpha)/lambda-H)/d);
}
// Candidate-rating fit: use historical AEe target as a working experiment;
// saturation occurs when a position's legal move losses cannot reach target.
static Law fitLaw(const Vec&d,int elo,double consistency=.50){
    double target=targetError(elo);
    // Wide bracket: at high s the law approaches uniform choice. The
    // target is capped only by the position's attainable uniform error.
    double l=1.e-8,r=1.e10;
    for(int j=0;j<64;++j){double mid=std::sqrt(l*r);
        Law m=choice(d,mid,consistency);
        if(expectedError(m.p,d)>target)r=mid;else l=mid;}
    return choice(d,std::sqrt(l*r),consistency);
}
struct Observation {
    double entropy=0,expectedLoss=0,actualLoss=0,expectedQuality=0,
           qualityVariance=0,centered=0,reganIPR=0,
           moveKL=0,thetaFisher=0,effFisher=0,
           evidence=0,localEvidence=0,score=0,
           predictedTimeSec=0,timeLogLikelihood=0,standardizedResidual=0;
    int topFirstVisibleDepth=-1;
    bool hasTime=false,possibleModelCrossing=false;
};
struct Tracker {
    long long moves=0;double sumExpectedLoss=0,sumLoss=0,sumCentered=0,sumVariance=0,
          logEvidence=0,localEvidence=0,sumTimingLog=0;
    // Sections 1.4, 1.5 and B.2 use unit-variance residuals, never raw
    // win-probability differences. Zero-variance positions are excluded.
    std::array<double,4> lagNumerator{};
    std::array<double,4> recentResidual{};
    int recentResidualCount=0;
    long long residualCount=0;
    std::array<long long,HE_MAX_TRACKED_DEPTH> depthCount{};
    std::array<long long,HE_MAX_TRACKED_DEPTH> depthTopCount{};
    std::array<double,HE_MAX_TRACKED_DEPTH> depthLoss{};
    long long timingSamples=0;
    Observation latest;
    void record(const Law&law,const Vec&alt,const Vec&quality,int chosen,
                double observedTimeSec,double expectedTimeSec,int firstTopDepth,
                double motor=.12){
        if(chosen<0||size_t(chosen)>=law.p.size())return;
        latest=Observation{};latest.entropy=entropy(law.p);
        latest.expectedLoss=expectedError(law.p,law.d);latest.actualLoss=law.d[chosen];
        latest.expectedQuality=mean(law.p,quality);latest.qualityVariance=variance(law.p,quality);
        latest.centered=quality[chosen]-latest.expectedQuality;
        latest.topFirstVisibleDepth=firstTopDepth;
        latest.reganIPR=intrinsicRating(latest.expectedLoss);
        latest.moveKL=kl(alt,law.p);latest.thetaFisher=latest.qualityVariance;
        Vec st=scoreTilt(quality,law.p),sr=scaleScore(law);
        latest.effFisher=fisher(law.p,sr,st).effective();
        latest.predictedTimeSec=expectedTimeSec;
        double step=safeLog(alt[chosen])-safeLog(law.p[chosen]);
        logEvidence=appendLogEvidence(logEvidence,step);
        latest.evidence=logEvidence;
        latest.localEvidence=localLogLR(.35,latest.centered,latest.qualityVariance);
        localEvidence+=latest.localEvidence;
        ++moves;sumExpectedLoss+=latest.expectedLoss;sumLoss+=latest.actualLoss;
        sumCentered+=latest.centered;sumVariance+=latest.qualityVariance;
        latest.score=standardizedScore(sumCentered,sumVariance);
        if(firstTopDepth>=1&&firstTopDepth<HE_MAX_TRACKED_DEPTH){
            ++depthCount[firstTopDepth];
            if(chosen==0)++depthTopCount[firstTopDepth];
            depthLoss[firstTopDepth]+=law.d[chosen];
        }
        if(latest.qualityVariance>1.e-15){
            const double e=latest.centered/std::sqrt(latest.qualityVariance);
            latest.standardizedResidual=e;
            for(int k=1;k<=recentResidualCount;++k)
                lagNumerator[k-1]+=e*recentResidual[k-1];
            for(int k=3;k>0;--k)recentResidual[k]=recentResidual[k-1];
            recentResidual[0]=e;
            recentResidualCount=std::min(4,recentResidualCount+1);
            ++residualCount;
        }
        if(observedTimeSec>motor){
            double m=std::log(std::max(expectedTimeSec-motor,.01))-.5*sq(.55);
            latest.timeLogLikelihood=logNormalLogDensity(observedTimeSec,motor,m,.55);
            latest.hasTime=true;++timingSamples;sumTimingLog+=latest.timeLogLikelihood;
        }
        // Only meaningful with externally calibrated H0; never label bot as human.
        latest.possibleModelCrossing=crossed(logEvidence,0.01);
    }
    double observedIPR()const{return moves?intrinsicRating(sumLoss/moves):NAN;}
    double expectedIPR()const{return moves?intrinsicRating(sumExpectedLoss/moves):NAN;}
    double lag(int k)const{return k>=1&&k<=4&&residualCount>k?
        lagNumerator[k-1]/residualCount:0;}
    double lagOne()const{return lag(1);}
    double portmanteau()const{
        double q=0;
        for(int k=1;k<=4&&residualCount>k;++k)
            q+=sq(lagNumerator[k-1])/(residualCount-k);
        return q;
    }
};
} // namespace HEMath

struct HEClockTiming {
    bool active=false;
    double motor=.12,spread=.55,baseMean=0,altMean=0;
};
static HEClockTiming HEPendingTime;
static HEMath::Tracker HEStats;
static void HEStatsReset(void){HEStats=HEMath::Tracker{};HEPendingTime=HEClockTiming{};}

// ===========================================================================
// MARK: - Section 1 equation helpers for the 400-1319 band
// ===========================================================================

// H.8: Lapse model — mix baseline with human-prior lapse (not uniform random)
// Uniform noise hangs queens and creates zero-intuition moves that flag
// anti-cheat.  Instead lapse toward the base law (pPrior) and block any
// lapse into a move that drops more than safeWinLoss in win probability.
static HEMath::Vec HEApplyLapse(const HEMath::Vec &p,
                                const HEMath::Vec &pPrior,
                                const HEMath::Vec &winLoss,
                                double epsilon) {
    if (epsilon <= 0 || p.empty()) return p;
    epsilon = HEMath::clamp(epsilon, 0.0, 0.03);
    const double safeWinLoss = 15.0; // never lapse into >15 pp win-prob drop
    HEMath::Vec out(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        double lapseMass = (winLoss[i] < safeWinLoss) ? pPrior[i] : 0.0;
        out[i] = (1.0 - epsilon) * p[i] + epsilon * lapseMass;
    }
    return HEMath::normalize(out);
}

// H.9: Stochastic depth mixture — geometric distribution over depths
// Returns weights πd(ρ) = (1-ρ)ρ^(d-1) / (1-ρ^D) for d=1..D
static HEMath::Vec HEDepthWeights(double rho, int D) {
    if (D <= 1 || rho <= 0) return {1.0};
    rho = HEMath::clamp(rho, 0.01, 0.99);
    HEMath::Vec w(D);
    double norm = 1.0 - std::pow(rho, D);
    for (int d = 0; d < D; ++d)
        w[d] = (1.0 - rho) * std::pow(rho, d) / norm;
    return HEMath::normalize(w);
}

// H.10: Latent attention state update (2-state Markov: focused/distracted)
// Updates the focused probability after observing a move quality
static void HEAttentionUpdate(double moveLoss) {
    double pF = HES.attentionFocused;
    double pD = 1.0 - pF;
    // Predict: apply transition matrix
    double predF = pF * (1.0 - HES.transitionToDistracted) + pD * HES.transitionToFocused;
    double predD = pF * HES.transitionToDistracted + pD * (1.0 - HES.transitionToFocused);
    // Emission likelihood ratio: focused players make smaller losses
    double emitF = std::exp(-moveLoss / 8.0);
    double emitD = std::exp(-moveLoss / 25.0);
    // Update: Bayes rule
    double nF = predF * emitF;
    double nD = predD * emitD;
    double total = nF + nD;
    if (total > 0) HES.attentionFocused = nF / total;
    else HES.attentionFocused = 0.5;
}

// H.13: Within-session sensitivity drift
// Returns a multiplier on s based on fatigue and error reaction
static double HESensitivityDrift(int ply) {
    double drift = 0.0;
    // Fatigue: sensitivity worsens as game progresses
    if (HES.sessionLength > 0)
        drift += HEHP.zetaFatigue * ((double)HES.sessionMoves / HES.sessionLength);
    // Error reaction: sensitivity worsens after a large loss
    if (HES.lastLoss > HEHP.errorThreshold)
        drift += HEHP.zetaError;
    return std::exp(drift);
}

// H.19: Risk attitude — mean-variance utility with position-dependent risk
// When ahead (bestCp >= 150): force positive rhoR (risk-averse, prefer safe)
// and add a simplification bonus for captures that trade material when winning.
// This makes the bot willingly accept small engine "inaccuracies" to simplify
// into trivially winning endgames — a strong human behavioral signature.
static HEMath::Vec HERiskTilt(const HEMath::Vec &quality, const HEMath::Vec &p,
                              double rhoR, int bestCp,
                              const std::vector<HEFeatures> &features, int n) {
    if (quality.empty()) return quality;
    // When winning, always risk-averse (positive rhoR → penalize variance)
    if (bestCp >= 150)
        rhoR = std::max(rhoR, 0.15 + 0.20 * HEMath::clamp((bestCp - 150) / 400.0, 0.0, 1.0));
    HEMath::Vec u(quality.size());
    for (size_t i = 0; i < (size_t)std::min((int)quality.size(), n); ++i) {
        double var_i = quality[i] * (1.0 - quality[i]);
        u[i] = -0.5 * rhoR * var_i;
        // Simplification bonus: captures that trade when ahead >= 2 pawns
        if (bestCp >= 200 && i < (size_t)n && features[i].isCapture
            && features[i].capturedValue >= 300)
            u[i] += 0.08;
    }
    // Fill any remaining entries
    for (size_t i = n; i < quality.size(); ++i) {
        double var_i = quality[i] * (1.0 - quality[i]);
        u[i] = -0.5 * rhoR * var_i;
    }
    return u;
}

// H.20: Playing style as feature tilt — exponential family
// Applies γ⊤φ(x,a) tilt with human-plausibility heuristics:
// - Bonus for captures, checks, natural recaptures
// - Bonus for quiet simplifying moves when ahead
// - Penalty for engine-only quiet moves (no human would find these)
static HEMath::Vec HEStyleTilt(const HEMath::Vec &p,
                                const std::vector<HEFeatures> &features,
                                int n) {
    if (n <= 0 || p.empty()) return p;
    HEMath::Vec scores(n);
    for (int i = 0; i < n; ++i) {
        double phi = 0;
        if (features[i].isCapture || features[i].givesCheck)
            phi += HEHP.gammaForce;
        double var_i = 0.5 * (1.0 - 0.5);
        phi += HEHP.gammaRisk * var_i;
        if (features[i].isQuiet && !features[i].isCapture)
            phi += HEHP.gammaSimp;
        // Penalize engine-only quiet moves: a quiet non-capture, non-check
        // move that only looks good at deep search is a massive anti-cheat
        // red flag.  Real humans play natural developing/consolidating moves.
        if (features[i].isQuiet && !features[i].isCapture
            && !features[i].givesCheck && !features[i].isPawnPush) {
            // Quiet retreats and lateral shuffles get a slight penalty
            // unless they're the top engine move (which may be natural)
            if (i > 0) phi -= 0.08;
        }
        scores[i] = phi;
    }
    return HEMath::tilt(p, scores, 1.0);
}

// H.21: Position classification for strength-by-type
enum HEPositionType { HE_POS_QUIET = 0, HE_POS_TACTICAL, HE_POS_ENDGAME };
static HEPositionType HEClassifyPosition(const HEBoard *b, const HEMath::Vec &delta, int n) {
    if (n >= 2 && delta[1] > 1.25) return HE_POS_TACTICAL;
    if (!b) return HE_POS_QUIET;
    int totalMaterial = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int v = HEPieceValue(b->sq[sq]);
        if (v >= 300 && v <= 400) totalMaterial += 3;
        else if (v == 500) totalMaterial += 5;
        else if (v == 900) totalMaterial += 9;
    }
    if (totalMaterial <= 16) return HE_POS_ENDGAME;
    return HE_POS_QUIET;
}

// H.21: Sensitivity shift by position type
static double HEPositionTypeShift(HEPositionType posType) {
    switch (posType) {
        case HE_POS_TACTICAL: return HEHP.etaTactical;
        case HE_POS_ENDGAME:  return HEHP.etaEndgame;
        default:              return 0.0;
    }
}

// H.24: Clock management — per-move budget
static double HEMoveBudget(int moveNumber, double clockRemaining, double increment) {
    double mrem = std::max(HEHP.clockMmin, HEHP.clockM0 - HEHP.clockM1 * moveNumber);
    return clockRemaining / mrem + increment;
}

// ===========================================================================
// MARK: - Move choice
// ===========================================================================


static HELine HEL[HE_MAX_PV];
static char     HEChosen[8];
static char     HEPdfChosen[8];

// PDF Sections 1.3-1.5 and 1.8: observable checks of the four intermediate
// presets. These are diagnostics, not extra move biases or Elo calibration.
struct HEPdfDiagnostics {
    int elo=-1,lastPly=-1,lastFirstVisibleDepth=-1;
    long long moves=0,residuals=0,saturated=0;
    double expectedLoss=0,observedLoss=0,centered=0,variance=0;
    double lagOneNumerator=0,previousResidual=0,lastResidual=0;
    double lastTarget=0,lastAttainable=0;
    std::array<long long,HE_MAX_TRACKED_DEPTH> depthCount{};
    std::array<long long,HE_MAX_TRACKED_DEPTH> depthTopCount{};
    std::array<double,HE_MAX_TRACKED_DEPTH> depthLoss{};
    void begin(int requestedElo,int ply) {
        if(elo!=requestedElo || (lastPly>=0 && ply>=0 && ply<lastPly))
            *this=HEPdfDiagnostics{};
        elo=requestedElo;
        lastPly=ply;
    }
    void record(const HEMath::Law&law,int selected,int firstVisibleDepth,
                double target,double attainable) {
        const double topProbability=law.p[0];
        const double topChosen=selected==0 ? 1.0 : 0.0;
        const double moveVariance=topProbability*(1.0-topProbability);
        ++moves;
        expectedLoss+=HEMath::expectedError(law.p,law.d);
        observedLoss+=law.d[selected];
        centered+=topChosen-topProbability;
        variance+=moveVariance;
        lastFirstVisibleDepth=firstVisibleDepth;
        lastTarget=target;
        lastAttainable=attainable;
        if(attainable+1.e-9<target)++saturated;
        if(moveVariance>1.e-15) {
            lastResidual=(topChosen-topProbability)/std::sqrt(moveVariance);
            if(residuals>0)lagOneNumerator+=lastResidual*previousResidual;
            previousResidual=lastResidual;
            ++residuals;
        }
        if(firstVisibleDepth>=1 && firstVisibleDepth<HE_MAX_TRACKED_DEPTH) {
            ++depthCount[firstVisibleDepth];
            depthTopCount[firstVisibleDepth]+=selected==0;
            depthLoss[firstVisibleDepth]+=law.d[selected];
        }
    }
    double expectedIPR()const {return moves?HEMath::intrinsicRating(expectedLoss/moves):NAN;}
    double observedIPR()const {return moves?HEMath::intrinsicRating(observedLoss/moves):NAN;}
    double lagOne()const {return residuals>1?lagOneNumerator/residuals:0.0;}
    double scoreZ()const {return HEMath::standardizedScore(centered,variance);}
};
static HEPdfDiagnostics HEPdfStats;


static int HEClassOf(double lossW) {
    for (int c = HE_NCLASS - 1; c > 0; c--) if (lossW >= HEClassLo[c]) return c;
    return HE_TOP;
}


static const char *HECommit(int i, double lossW) {
    HELastOwnTo       = HEParseMove(HEL[i].move).to;
    HES.lastCp        = HEL[i].cp;
    HES.haveLastCp = 1;
    HES.ownMoves++;
    HES.count[HEClassOf(lossW)]++;
    memcpy(HEChosen, HEL[i].move, sizeof(HEChosen));
    return HEChosen;
}


// Returns the move to play, or NULL when outside the human band.
// Positions must be searched to completion using MultiPV; this ONLY selects
// among returned Stockfish root moves. Unreported legal moves are out of scope.
static const char *HECoreChooseMove(const char *fen, int ply, int elo) {
    if (!HEHumanBand(elo) || HELineCount<1) return NULL;
    static const bool solverOK=HEMath::referenceSolverCheck();
    if(!solverOK)NSLog(@"[ChessAssistantPro] Reference move-law solver check failed");
    HEHP=HEParamsFor(elo);
    if(!HES.init)HECoreNewGame((ply%2)==0,elo);

    int widest=0;
    for(int depth=1;depth<HE_MAX_TRACKED_DEPTH;++depth)
        widest=std::max(widest,HEDepthCounts[depth]);
    // At 400, keep about 82% of the widest candidate set before accepting a
    // deeper partial search; the fraction tapers to 50% at 1319. An absolute
    // floor of 9 -> 5 covers positions with only a few reported legal moves.
    // If fewer lines exist, use the widest available set.
    int needed=std::min(widest,std::max((int)lround(8.0-3.0*HEHP.t),
        (int)lround(widest*(.75-.25*HEHP.t))));
    // H.9: stochastic depth selection — geometric mixture πd(ρ)
    // Collect all complete depths, then sample from geometric weights
    std::vector<int> completedDepths;
    for(int depth=1;depth<HE_MAX_TRACKED_DEPTH;++depth)
        if(HEDepthCounts[depth]>=needed&&HEDepthCounts[depth]>0)
            completedDepths.push_back(depth);
    int chosenDepth=-1;
    if(!completedDepths.empty()){
        int D=(int)completedDepths.size();
        HEMath::Vec dw=HEDepthWeights(HEHP.depthRho,D);
        // Weighted random selection among available depths
        double r=HEFrand(),acc=0;
        for(int i=D-1;i>=0;--i){
            acc+=dw[i];
            if(r<acc||i==0){chosenDepth=completedDepths[i];break;}
        }
    }
    // Never combine evaluations from different depths in the probability law.
    int n=0;
    if(chosenDepth>=0){
        for(int i=0;i<HE_MAX_PV;++i)
            if(HELinesAtDepth[chosenDepth][i].valid)
                HEL[n++]=HELinesAtDepth[chosenDepth][i];
    } else {
        int bestI=-1;
        for(int i=0;i<HELineCount;++i)if(HELines[i].valid&&
           (bestI<0||HELines[i].depth>HELines[bestI].depth))bestI=i;
        if(bestI>=0)HEL[n++]=HELines[bestI];
    }
    if(n==0)return NULL;
    std::sort(HEL,HEL+n,[](const HELine&a,const HELine&b){return a.cp>b.cp;});
    // Section 1.8: first depth where the final-depth top move was itself the
    // top-scored move in a snapshot with at least two reported candidates.
    int topFirstDepth=-1;
    if(chosenDepth>=1&&n>1){
        for(int depth=1;depth<=chosenDepth;++depth){
            if(HEDepthCounts[depth]<2)continue;
            const HELine *top=nullptr;
            for(int i=0;i<HE_MAX_PV;++i){
                const HELine *line=&HELinesAtDepth[depth][i];
                if(line->valid&&(!top||line->cp>top->cp))top=line;
            }
            if(top&&strcmp(top->move,HEL[0].move)==0){topFirstDepth=depth;break;}
        }
    }

    HEBoard board{};
    bool haveBoard=HEParseFEN(fen,&board)!=0;
    int side=haveBoard?board.whiteToMove:HES.isWhite;
    HEMath::Vec values(n), quality(n), proposedMix(HE_NCLASS), winLoss(n);
    std::vector<HEFeatures> features(n);
    double bestWin=HEWinPct(HEL[0].cp);
    int jump=HES.haveLastCp?HEL[0].cp-HES.lastCp:0;
    bool mateNow=HEL[0].mateIn>0&&HEL[0].mateIn<=3;
    bool opportunity=mateNow || (HES.haveLastCp&&jump>=250&&HEL[0].cp>=200
                 &&HES.lastCp<HEHP.convThresh);

    for(int i=0;i<n;++i){
        values[i]=std::max(-2000.0,std::min(2000.0,(double)HEL[i].cp))/100.0;
        quality[i]=HEWinPct(HEL[i].cp)/100.0;
        winLoss[i]=std::max(0.0,bestWin-HEWinPct(HEL[i].cp));
        if(haveBoard)features[i]=HEExtract(&board,HEParseMove(HEL[i].move));
        else memset(&features[i],0,sizeof(HEFeatures));
    }
    // Eq 01-04, 10-11, 73-75: position-specific root equation and fit to the
    // reference's extrapolated rating/expected-loss scale (not a measured Elo).
    HEMath::Vec delta=HEMath::losses(values);

    // H.21: classify position and shift sensitivity by type
    HEPositionType posType = haveBoard ? HEClassifyPosition(&board, delta, n)
                                       : HE_POS_QUIET;
    double etaShift = HEPositionTypeShift(posType);

    // H.13: within-session sensitivity drift (fatigue + error reaction)
    double driftMult = HESensitivityDrift(ply);

    // H.10: attention-weighted sensitivity — blend focused and distracted
    double attnSigma = HES.attentionFocused * HES.focusedSigma
                     + (1.0 - HES.attentionFocused) * HES.distractedSigma;

    // H.27: composite sensitivity: s_eff = s(R) * exp(η_κ) * drift * σ_attention
    // fitLaw finds the s that matches the target AEe; we scale its result
    HEMath::Law law=HEMath::fitLaw(delta,elo,.50);
    double sComposite = law.s * std::exp(etaShift) * driftMult * attnSigma;
    sComposite = HEMath::clamp(sComposite, law.s * 0.5, law.s * 3.0);
    if (std::fabs(sComposite - law.s) > 1e-10)
        law = HEMath::choice(delta, sComposite, law.c);

    // Eq 61-63, 68-69: a CLOCK-CONDITIONED regime changes sensitivity.
    // Larger s = weaker play.  More budget should LOWER s (stronger),
    // low clock should RAISE s (weaker / time-trouble degradation).
    if(HEClockSeconds>0){
        double budget = HEMoveBudget(ply / 2, HEClockSeconds, 0);
        double clockScale=HEMath::clamp(
            1.0-0.10*std::log(budget / (HEClockSeconds / 40.0 + 1.0)),0.85,1.35);
        if(HEClockSeconds<30.0){
            clockScale*=1.0+0.45*(1.0-HEClockSeconds/30.0);
        }
        law=HEMath::choice(delta,law.s*clockScale,law.c);
    }

    // H.19: risk-adjusted quality — risk-averse when ahead, simplify trades
    HEMath::Vec riskQ = HERiskTilt(quality, law.p, HEHP.riskAversion,
                                   HEL[0].cp, features, n);

    // Model the alternative's intent to win through a positive exponential
    // tilt (Eq 17-20) using risk-adjusted quality.
    double theta=.60+0.75*HEHP.t;
    HEMath::Vec winning=HEMath::tilt(law.p,riskQ,theta);

    // H.20: apply playing style as feature tilt
    winning = HEStyleTilt(winning, features, n);

    double cx=haveBoard?HEComplexity(&board,side):1.0;
    double sharp=std::min(1.5,std::max(.70,.60+.40*cx));
    for(int c=0;c<HE_NCLASS;++c)proposedMix[c]=HEHP.mix[c];
    proposedMix[HE_MISTAKE]*=sharp;proposedMix[HE_BLUNDER]*=sharp;

    // When winning (bestQuality >= 0.60 i.e. eval >= +1.5 pawns or so),
    // suppress blunders entirely and reduce mistakes heavily.
    // Real strong humans beat 400-1319 by keeping blunders near zero while
    // still making natural inaccuracies (15-60 cp).
    bool isAhead = (quality[0] >= 0.60);
    if(isAhead){
        proposedMix[HE_BLUNDER]=0.0;
        proposedMix[HE_MISTAKE]*=0.25;
    }
    if(ply<8){
        proposedMix[HE_MISTAKE]*=.85-.25*HEHP.t;
        proposedMix[HE_BLUNDER]*=.75-.30*HEHP.t;
    }
    if(HES.haveLastCp&&HES.lastCp>=HEHP.convThresh&&HEL[0].cp>=HEHP.convThresh){
        proposedMix[HE_MISTAKE]*=.90-.20*HEHP.t;
        proposedMix[HE_BLUNDER]*=0.0;  // zero blunders while converting
    }
    proposedMix=HEMath::normalize(proposedMix);

    // Actual available quality classes are constrained by Stockfish's finite
    // MultiPV list; unrepresented classes are not fabricated.
    HEMath::Vec classMass(HE_NCLASS,0),candidate(n,0);
    int available[HE_NCLASS]={0};
    for(int i=0;i<n;++i){
        int c=HEClassOf(winLoss[i]);
        classMass[c]+=winning[i];
        ++available[c];
    }
    for(int i=0;i<n;++i){
        int c=HEClassOf(winLoss[i]);
        double factor=proposedMix[c]/std::max(classMass[c],HEMath::tiny);
        candidate[i]=winning[i]*factor*HEHumanBias(&features[i],ply);
        // Low-rating visual mistakes remain possible, but a bot trying to win
        // does not deliberately choose a forced losing mate on every cycle.
        if(HEL[i].mateIn<0&&HEL[0].mateIn>=0)candidate[i]*=.10+.15*(1.0-HEHP.t);
        if(c>=HE_MISTAKE&&features[i].visibleMistake)candidate[i]*=HEHP.visW;
    }

    if(opportunity&&n>1){
        // Always spot obvious: mate-in-1/2, hanging pieces (free captures),
        // and basic recaptures. These are moves any human sees instantly.
        bool obviousWin = (HEL[0].mateIn>0 && HEL[0].mateIn<=2)
            || (features[0].isCapture && features[0].capturedValue>=300
                && winLoss[0]<2.0);
        if(obviousWin){
            // Suppress blunders in obvious positions — human always takes
            for(int i=0;i<n;++i)if(winLoss[i]>5)candidate[i]*=.15;
            HES.missStreak=0;
        }else{
            // Non-obvious tactics: roll for recognition
            double spot=HEHP.spotTactic;
            spot=HEMath::clamp(spot+.10*HES.missStreak,.02,.995);
            bool saw=HEFrand()<spot;
            if(saw){
                for(int i=0;i<n;++i)if(winLoss[i]>5)candidate[i]*=.20;
                HES.missStreak=0;
            }else{
                // When tactic is missed: do NOT suppress good solid moves.
                // Only slightly boost quiet developing/consolidating moves
                // so the bot plays naturally instead of blundering.
                for(int i=0;i<n;++i){
                    if(winLoss[i]<2.0 && features[i].isQuiet)
                        candidate[i]*=1.15;
                }
            }
        }
    }

    HEMath::Vec selectedLaw=HEMath::normalize(candidate);

    // H.8: lapse model — mix with human-prior (not uniform), gated by safety
    selectedLaw = HEApplyLapse(selectedLaw, law.p, winLoss, HEHP.lapseRate);

    int k=HEPick(selectedLaw.data(),n);
    if(k<0)k=0;
    NSLog(@"[ChessAssistantPro] Human choice ELO=%d ply=%d depth=%d candidates=%d classes=%d/%d/%d/%d/%d picked=%d loss=%.1fpp attn=%.2f drift=%.3f eta=%.3f",
          elo,ply,chosenDepth,n,available[0],available[1],available[2],
          available[3],available[4],HEClassOf(winLoss[k]),winLoss[k],
          HES.attentionFocused,driftMult,etaShift);
    if(opportunity&&k!=0){HES.misses++;HES.missStreak++;}
    else if(opportunity)HES.missStreak=0;

    // H.10: update attention state based on the move's quality
    HEAttentionUpdate(winLoss[k]);
    // H.13: track session state for next move's drift calculation
    HES.lastLoss = winLoss[k];
    HES.sessionMoves++;

    // 05-08, 17-63: observe the actually selected quality and update the
    // diagnostics under the reference and the actual human-biased policy.
    // Context-gated timing: fast-path for openings, forced moves, and time
    // scramble; reserve long thinks for genuinely complex positions.
    double suggested=0;
    HEPendingTime=HEClockTiming{};
    if(HEClockSeconds>0){
        double budget = HEMoveBudget(ply / 2, HEClockSeconds, 0);
        double ent=HEMath::entropy(selectedLaw);
        double logM=HEMath::timeLogMean(-.9,.30,.10,ent,budget);
        HEPendingTime.active=true;
        HEPendingTime.baseMean=HEMath::timeLogMean(-.9,.30,.10,HEMath::entropy(law.p),budget);
        HEPendingTime.altMean=logM;
        suggested=.12+std::exp(logM+.5*.55*.55);

        // Context gating: override the base timing for specific scenarios
        bool isOpening  = (ply < 10);
        bool isForced   = (n == 1) || (n >= 2 && winLoss[1] > 20.0
                          && features[0].isCapture);
        bool isComplex  = (ent > 1.5) || (n >= 2 && delta[1] > 1.0);
        bool isScramble = (HEClockSeconds < 20.0);

        if(isOpening && !isComplex){
            // Opening book / developing moves: short, natural delay
            suggested=0.4+HEFrand()*2.1;
        }else if(isForced){
            // Only legal move or obvious recapture: near-instant
            suggested=0.3+HEFrand()*1.1;
        }else if(isScramble){
            // Low-clock scramble: fast premove pacing
            suggested=0.2+HEFrand()*1.3;
        }else if(isComplex){
            // High-entropy critical junction: think longer
            double gap=0;
            if(delta.size()>1)gap=*std::min_element(delta.begin()+1,delta.end());
            double attnB = 0.8 * attnSigma;
            HEMath::Diffusion d=HEMath::diffusion(ent,budget,
                                          gap,law.s,attnB,.24,.04,.12,.35);
            double diffusionSec=HEMath::expectedExit(d);
            if(std::isfinite(diffusionSec)&&diffusionSec>0)
                suggested=0.4*suggested+0.6*diffusionSec;
            suggested=HEMath::clamp(suggested,2.0,std::min(25.0,HEClockSeconds*0.20));
        }else{
            // Routine quiet / developing move: standard timing
            double gap=0;
            if(delta.size()>1)gap=*std::min_element(delta.begin()+1,delta.end());
            double attnB = 0.8 * attnSigma;
            HEMath::Diffusion d=HEMath::diffusion(ent,budget,
                                          gap,law.s,attnB,.24,.04,.12,.35);
            double diffusionSec=HEMath::expectedExit(d);
            if(std::isfinite(diffusionSec)&&diffusionSec>0)
                suggested=.5*suggested+.5*diffusionSec;
        }
        suggested=HEMath::clamp(suggested,.15,std::max(.15,HEClockSeconds*.15));
    }
    HEStats.record(law,selectedLaw,quality,k,-1,suggested,topFirstDepth);
    if(HEStats.residualCount>=5&&HEStats.residualCount%10==0)
        NSLog(@"[ChessAssistantPro] Uncalibrated move diagnostics n=%lld lag1=%.3f Q4=%.3f topFirstDepth=%d",
              HEStats.residualCount,HEStats.lagOne(),HEStats.portmanteau(),topFirstDepth);
    return HECommit(k,winLoss[k]);
}

// Sections 1.1-1.5, 1.8 and 8.1-8.4 of the PDF: sample the move law and
// measure its predicted/observed errors, residuals and depth strata.
// The Elo-to-error target in fitLaw is historical extrapolation, not a
// measured Stockfish 19 rating.
// Only one completed MultiPV depth is used; unreported legal moves cannot be
// assigned a probability from the available UCI output.
static const char *HEPdfCoreChooseMove(int elo,int ply) {
    if(!HEPdfBand(elo))return nullptr;
    HEPdfStats.begin(elo,ply);
    int widest=0;
    for(int d=1;d<HE_MAX_TRACKED_DEPTH;++d)
        widest=std::max(widest,HEDepthCounts[d]);
    if(widest<2)return nullptr;
    const int needed=std::max(2,(3*widest+3)/4);
    int depth=-1;
    for(int d=HE_MAX_TRACKED_DEPTH-1;d>=1;--d)
        if(HEDepthCounts[d]>=needed){depth=d;break;}
    if(depth<0)return nullptr;
    int n=0;
    for(int i=0;i<HE_MAX_PV;++i)
        if(HELinesAtDepth[depth][i].valid)
            HEL[n++]=HELinesAtDepth[depth][i];
    if(n<2)return nullptr;
    std::sort(HEL,HEL+n,[](const HELine &a,const HELine &b){return a.cp>b.cp;});
    HEMath::Vec values(n);
    for(int i=0;i<n;++i)
        values[i]=std::max(-20.0,std::min(20.0,HEL[i].cp/100.0));
    const HEMath::Vec losses=HEMath::losses(values);
    const HEMath::Law law=HEMath::fitLaw(losses,elo,.50);
    const int selected=HEPick(law.p.data(),n);
    if(selected<0)return nullptr;
    int firstVisibleDepth=-1;
    for(int d=1;d<=depth;++d)
        if(HEDepthCounts[d]>=2 && HELinesAtDepth[d][0].valid &&
           strcmp(HELinesAtDepth[d][0].move,HEL[0].move)==0) {
            firstVisibleDepth=d;
            break;
        }
    const double target=HEMath::targetError(elo);
    const double attainable=HEMath::total(losses)/n;
    HEPdfStats.record(law,selected,firstVisibleDepth,target,attainable);
    memcpy(HEPdfChosen,HEL[selected].move,sizeof(HEPdfChosen));
    NSLog(@"[ChessAssistantPro] PDF move law preset=%d depth=%d candidates=%d s=%.5f c=%.3f p0=%.4f selected=%s targetAEe=%.4f attainableAEe=%.4f",
          elo,depth,n,law.s,law.c,law.p0,HEPdfChosen,target,attainable);
    if(HEPdfStats.moves%10==0)
        NSLog(@"[ChessAssistantPro] PDF diagnostics preset=%d moves=%lld expectedIPR=%.1f observedIPR=%.1f topScoreZ=%.3f residualLag1=%.3f saturated=%lld firstTopDepth=%d",
              elo,HEPdfStats.moves,HEPdfStats.expectedIPR(),HEPdfStats.observedIPR(),
              HEPdfStats.scoreZ(),HEPdfStats.lagOne(),HEPdfStats.saturated,
              firstVisibleDepth);
    return HEPdfChosen;
}

// Optional caller-supplied clock, not a fabricated game clock. User controls
// clock/observed timings from the bot match controller.
void HEClockRemainingSeconds(double seconds){
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    HEClockSeconds=(std::isfinite(seconds)&&seconds>0)?seconds:-1;
}
void HERecordDecisionTimeSeconds(double seconds){
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    if(HEPendingTime.active&&std::isfinite(seconds)&&seconds>HEPendingTime.motor){
        double l0=HEMath::logNormalLogDensity(seconds,HEPendingTime.motor,
                                    HEPendingTime.baseMean,HEPendingTime.spread);
        double l1=HEMath::logNormalLogDensity(seconds,HEPendingTime.motor,
                                    HEPendingTime.altMean,HEPendingTime.spread);
        // Eq 23-28: joint move+time log evidence and conditional KL.
        HEStats.logEvidence+=l1-l0;
        HEStats.latest.evidence=HEStats.logEvidence;
        HEStats.latest.hasTime=true;
        HEStats.latest.timeLogLikelihood=l0;
        HEStats.sumTimingLog+=l0;++HEStats.timingSamples;
        HEPendingTime.active=false; // a timing observation is applied once
    }
}
// One sample of time to optionally delay applying the chosen move in a bot
// controller. Search time is separate from move presentation/think time.
int HESuggestThinkTimeMilliseconds(){
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    if(!HEPendingTime.active)return 0;
    double u=std::max(1.0e-12,HEFrand());
    double z=std::sqrt(-2.0*std::log(u))*std::cos(2*HEMath::pi*HEFrand());
    double seconds=HEPendingTime.motor+
        std::exp(HEPendingTime.altMean+HEPendingTime.spread*z);
    seconds=HEMath::clamp(seconds,.12,std::max(.12,HEClockSeconds*.15));
    return static_cast<int>(std::lround(seconds*1000));
}
// Inspect actual source-derived statistics. These have no human-calibration
// meaning until fitted to a defined human rating pool and time control.
struct HEFormulaReport {
    double expectedIPR,observedIPR,expectedLoss,observedLoss,scoreZ,lagOne;
    double lagTwo,lagThree,lagFour,portmanteauQ4,standardizedResidual;
    double depthTopRate,depthMeanLoss;
    double moveLogEvidence,localLogEvidence,moveKLDivergence,effectiveInformation;
    double entropy,predictedThinkSeconds,timingLogLikelihood;
    double timingKLDivergence,expectedWinProbability,modelSampleLowerBound;
    long long moves,timingSamples,residualCount,depthStratumCount;
    int topFirstVisibleDepth;
};
HEFormulaReport HEGetFormulaReport(){
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    HEFormulaReport r{};
    r.expectedIPR=HEStats.expectedIPR();r.observedIPR=HEStats.observedIPR();
    r.moves=HEStats.moves;r.timingSamples=HEStats.timingSamples;
    if(r.moves){r.expectedLoss=HEStats.sumExpectedLoss/r.moves;
                r.observedLoss=HEStats.sumLoss/r.moves;}
    r.scoreZ=HEMath::standardizedScore(HEStats.sumCentered,HEStats.sumVariance);
    r.lagOne=HEStats.lagOne();r.lagTwo=HEStats.lag(2);
    r.lagThree=HEStats.lag(3);r.lagFour=HEStats.lag(4);
    r.portmanteauQ4=HEStats.portmanteau();
    r.standardizedResidual=HEStats.latest.standardizedResidual;
    r.residualCount=HEStats.residualCount;
    r.topFirstVisibleDepth=HEStats.latest.topFirstVisibleDepth;
    if(r.topFirstVisibleDepth>=1&&r.topFirstVisibleDepth<HE_MAX_TRACKED_DEPTH){
        const int d=r.topFirstVisibleDepth;
        r.depthStratumCount=HEStats.depthCount[d];
        if(r.depthStratumCount>0){
            r.depthTopRate=(double)HEStats.depthTopCount[d]/r.depthStratumCount;
            r.depthMeanLoss=HEStats.depthLoss[d]/r.depthStratumCount;
        }
    }
    r.moveLogEvidence=HEStats.logEvidence;
    r.localLogEvidence=HEStats.localEvidence;
    r.moveKLDivergence=HEStats.latest.moveKL;
    r.effectiveInformation=HEStats.latest.effFisher;
    r.entropy=HEStats.latest.entropy;
    r.predictedThinkSeconds=HEStats.latest.predictedTimeSec;
    r.timingLogLikelihood=HEStats.sumTimingLog;
    r.timingKLDivergence=HEPendingTime.active?
        HEMath::logNormalKL(HEPendingTime.altMean,HEPendingTime.baseMean,
                             HEPendingTime.spread):0.0;
    r.expectedWinProbability=HEStats.latest.expectedQuality;
    r.modelSampleLowerBound=HEStats.latest.moveKL>HEMath::tiny?
        HEMath::effectiveSampleLower(.01,.10,HEStats.latest.moveKL):INFINITY;
    return r;
}

// ===========================================================================
// MARK: - Objective-C API
// ===========================================================================


#ifdef __OBJC__


void HENewGame(BOOL playingWhite, int elo) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    HECoreNewGame(playingWhite ? 1 : 0, elo);
}


NSArray<NSString *> *HEConfigureCommands(int elo) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    char cmds[8][HE_CMD_LEN];
    int n = HECoreConfigure(elo, cmds, 8);
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:n];
    for (int i = 0; i < n; i++) [out addObject:[NSString stringWithUTF8String:cmds[i]]];
    return out;
}


NSString *HEChooseMove(const char *fen, int ply, int elo) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    const char *m = HECoreChooseMove(fen, ply, elo);
    return m ? [NSString stringWithUTF8String:m] : nil;
}

static NSString *HEPdfChooseMove(int elo,int ply) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    const char *move=HEPdfCoreChooseMove(elo,ply);
    return move ? [NSString stringWithUTF8String:move] : nil;
}


NSDictionary *HEDescribeElo(int elo) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    if (!HEHumanBand(elo)) {
        if(HEPdfBand(elo)) {
            const bool have=HEPdfStats.elo==elo && HEPdfStats.moves>0;
            const int depth=have?HEPdfStats.lastFirstVisibleDepth:-1;
            const long long depthN=depth>=1&&depth<HE_MAX_TRACKED_DEPTH?
                HEPdfStats.depthCount[depth]:0;
            return @{ @"mode": @"PDF candidate move law · uncalibrated ELO",
                      @"skill": @20,
                      @"moves": @(have?HEPdfStats.moves:0),
                      @"targetAEe": @(HEMath::targetError(elo)),
                      @"expectedAEe": @(have?HEPdfStats.expectedLoss/HEPdfStats.moves:0),
                      @"observedAEe": @(have?HEPdfStats.observedLoss/HEPdfStats.moves:0),
                      @"expectedIPR": @(have?HEPdfStats.expectedIPR():0),
                      @"observedIPR": @(have?HEPdfStats.observedIPR():0),
                      @"topMoveScoreZ": @(have?HEPdfStats.scoreZ():0),
                      @"topMoveResidualLag1": @(have?HEPdfStats.lagOne():0),
                      @"saturatedPositions": @(have?HEPdfStats.saturated:0),
                      @"firstTopVisibleDepth": @(depth),
                      @"depthStratumCount": @(depthN),
                      @"depthTopRate": @(depthN?(double)HEPdfStats.depthTopCount[depth]/depthN:0),
                      @"depthMeanLoss": @(depthN?HEPdfStats.depthLoss[depth]/depthN:0) };
        }
        return @{ @"mode"     : elo < 3190 ? @"Stockfish UCI ELO limit"
                                              : @"Full strength · no fixed ELO",
                     @"skill" : @20 };
    }
    HEParams p = HEParamsFor(elo);
    return @{
        @"mode"          : [NSString stringWithFormat:
                            @"Human sim · depth %d · MultiPV %d · blunder %.0f%%/move",
                            p.depth, p.multipv, p.mix[HE_BLUNDER] * 100.0],
        @"depth"         : @(p.depth),
        @"multipv"       : @(p.multipv),
        @"moves"         : @(HES.ownMoves),
         @"best"       : @(HES.count[HE_TOP]),
         @"good"       : @(HES.count[HE_GOOD]),
         @"inaccuracy" : @(HES.count[HE_INACC]),
         @"mistake"    : @(HES.count[HE_MISTAKE]),
         @"blunder"    : @(HES.count[HE_BLUNDER]),
         @"miss"       : @(HES.misses),
         @"expectedIPR": @(HEStats.expectedIPR()),
         @"observedIPR": @(HEStats.observedIPR()),
         @"moveEvidence": @(HEStats.logEvidence),
         @"moveScoreZ" : @(HEMath::standardizedScore(HEStats.sumCentered,HEStats.sumVariance)),
         @"standardizedResidual": @(HEStats.latest.standardizedResidual),
         @"residualCount": @(HEStats.residualCount),
         @"lag1": @(HEStats.lag(1)),
         @"lag2": @(HEStats.lag(2)),
         @"lag3": @(HEStats.lag(3)),
         @"lag4": @(HEStats.lag(4)),
         @"portmanteauQ4": @(HEStats.portmanteau()),
         @"topFirstVisibleDepth": @(HEStats.latest.topFirstVisibleDepth),
         @"predictedThinkSeconds": @(HEStats.latest.predictedTimeSec)
    };
}


#endif
#pragma clang diagnostic pop

// Preserve the existing FEN-transition observer for opponent recapture bias.
// Accept a transition only when exactly one move reproduces the next board.
// This also handles castling, en passant and promotions without guessing from
// a changed-square count.
static NSString *HEInferMoveBetweenBoards(const HEBoard *before,
                                          const HEBoard *after) {
    if (before->whiteToMove == after->whiteToMove) return nil;
    NSString *match = nil;
    for (int from = 0; from < 64; ++from) {
        char piece = before->sq[from];
        if (piece == '.' || HEIsWhitePiece(piece) != before->whiteToMove) continue;
        for (int to = 0; to < 64; ++to) {
            if (from == to) continue;
            BOOL promotion = tolower((unsigned char)piece) == 'p' &&
                HERank(to) == (before->whiteToMove ? 7 : 0);
            const char *promotions = promotion ? "qrbn" : "-";
            for (const char *promo = promotions; *promo; ++promo) {
                HEMove candidate = {from, to, (char)(promotion ? *promo : 0)};
                HEBoard result = HEApply(before, candidate);
                if (result.whiteToMove != after->whiteToMove ||
                    memcmp(result.sq, after->sq, sizeof(result.sq)) != 0) continue;
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

static NSString *HEInferOnePlyMove(const char *priorFen, const char *currentFen) {
    HEBoard prior, current;
    if (!HEParseFEN(priorFen, &prior) || !HEParseFEN(currentFen, &current)) return nil;
    return HEInferMoveBetweenBoards(&prior, &current);
}

// When only our own turns were observed, first apply the move recommended at
// the previous search, then infer the opponent's reply from the new position.
NSString *HEInferOpponentMove(const char *priorFen, const char *selectedMove,
                              const char *currentFen) {
    HEBoard prior, current;
    if (!HEParseFEN(priorFen, &prior) || !HEParseFEN(currentFen, &current)) return nil;
    HEMove ours = HEParseMove(selectedMove);
    if (ours.from < 0 || ours.to < 0 || prior.sq[ours.from] == '.' ||
        HEIsWhitePiece(prior.sq[ours.from]) != prior.whiteToMove ||
        current.whiteToMove != prior.whiteToMove) return nil;
    HEBoard afterOurs = HEApply(&prior, ours);
    return HEInferMoveBetweenBoards(&afterOurs, &current);
}


static void HEInvalidateConfiguration(void) {
    std::lock_guard<std::recursive_mutex> lock(HEMutex);
    HECfgElo = -1;
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
    BOOL pdfElo = stockfish19 && HEPdfBand(requestedElo);
    int ply = (lowElo || pdfElo) ? CAPFenPly(fen) : -1;
    NSString *humanGo = nil;
    if (lowElo) {
        BOOL newGame = humanLastElo != requestedElo || humanLastPly < 0 ||
            ply < humanLastPly ||
            (ply <= 1 && humanLastPly > 1);
        if (newGame) {
            HENewGame(strstr(fen, " b ") == nullptr, requestedElo);
            humanLastFen.clear(); humanLastMove.clear();
            NSLog(@"[ChessAssistantPro] Human band new game, ply=%d", ply);
        } else if (ply == humanLastPly + 1 && !humanLastFen.empty()) {
            HEBoard priorBoard;
            BOOL havePrior = HEParseFEN(humanLastFen.c_str(), &priorBoard);
            NSString *played = havePrior ?
                HEInferOnePlyMove(humanLastFen.c_str(), fen) : nil;
            BOOL opponentMoved = havePrior && priorBoard.whiteToMove != HES.isWhite;
            if (opponentMoved) {
                HENoteOpponentMove(played.UTF8String);
                NSLog(@"[ChessAssistantPro] Human band opponent move %@",
                      played ?: @"unavailable from one-ply FEN transition");
            } else if (played) {
                std::lock_guard<std::recursive_mutex> lock(HEMutex);
                HELastOwnTo = HEParseMove(played.UTF8String).to;
            }
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
        if (stockfish19) HEInvalidateConfiguration();
    }
    if (stockfish19) {
        // The engine process is reused, so slider and engine-mode transitions
        // must restore the corresponding UCI options.
        int uciElo = requestedElo;
        if (!lowElo && cachedThreads != 2) {
            if (!CAPWrite(input[1], "setoption name Threads value 2\n")) goto done;
            cachedThreads = 2;
        }
        if (lowElo) {
            for (NSString *command in HEConfigureCommands(requestedElo)) {
                if ([command hasPrefix:@"go "]) {
                    humanGo = command;
                } else if ([command isEqualToString:@"setoption name Threads value 1"]) {
                    if (cachedThreads != 1) {
                        if (!CAPWrite(input[1], "setoption name Threads value 1\n")) goto done;
                        cachedThreads = 1;
                    }
                } else if (!CAPWrite(input[1], std::string(command.UTF8String) + "\n")) goto done;
            }
            if (!humanGo) goto done;
            HEResetLines();
            NSLog(@"[ChessAssistantPro] Stockfish human band ELO=%d ply=%d %@",
                  requestedElo, ply, humanGo);
        } else if (pdfElo) {
            // Full-strength evaluations from one depth, sampled by the PDF
            // law. Never reuse the low-Elo or high-Elo UCI option state.
            HEInvalidateConfiguration();
            if (!CAPWrite(input[1], "setoption name UCI_LimitStrength value false\n") ||
                !CAPWrite(input[1], "setoption name Skill Level value 20\n") ||
                !CAPWrite(input[1], "setoption name MultiPV value 16\n")) goto done;
            HEResetLines();
            NSLog(@"[ChessAssistantPro] PDF move law preset=%d; searching 16 lines",requestedElo);
        } else {
            // Preserve the original 2000–3200 UCI path. Reset the sampled
            // modes' options if the slider later returns to either band.
            HEInvalidateConfiguration();
            if (!CAPWrite(input[1], "setoption name MultiPV value 1\n") ||
                !CAPWrite(input[1], "setoption name Skill Level value 20\n") ||
                !CAPWrite(input[1], std::string("setoption name UCI_LimitStrength value ") +
                                    (requestedElo >= 3190 ? "false\n" : "true\n"))) goto done;
            if (requestedElo < 3190 &&
                !CAPWrite(input[1], "setoption name UCI_Elo value " +
                                    std::to_string(uciElo) + "\n")) goto done;
            NSLog(@"[ChessAssistantPro] Stockfish preset=%d UCI_Elo=%@ MultiPV=1",
                  requestedElo, requestedElo >= 3190 ? @"full strength" : @(uciElo));
        }
    }
    if (!CAPWrite(input[1], "isready\n") || !CAPWaitForLine(output[0], "readyok", deadline)) goto done;
    if (!CAPWrite(input[1], std::string("position fen ") + fen + "\n")) goto done;
    if (!CAPWrite(input[1], lowElo ? std::string(humanGo.UTF8String) + "\n" :
                          (pdfElo ? "go depth 12 movetime 1200\n" :
                           (stockfish19 ? "go movetime 200\n" : "go depth 16\n")))) goto done;
    deadline = CFAbsoluteTimeGetCurrent() + (stockfish19 ? 3.0 : 15.0);
    while (CAPReadLine(output[0], line, deadline)) {
        if (line.find("CRITICAL ERROR") != std::string::npos) break;
        if (line.compare(0, 9, "bestmove ") != 0) {
            if (lowElo || pdfElo) HEParseInfo(line.c_str());
            continue;
        }
        size_t end = line.find(' ', 9);
        bestMove = line.substr(9, end == std::string::npos ? end : end - 9);
        success = CAPUciMove(bestMove);
        if (success && (lowElo || pdfElo)) {
            if (humanLastFen == fen && humanLastElo == requestedElo &&
                CAPUciMove(humanLastMove)) {
                bestMove = humanLastMove;
            } else {
                NSString *chosen = lowElo ?
                    HEChooseMove(fen, MAX(ply, 0), requestedElo) :
                    HEPdfChooseMove(requestedElo,ply);
                if (chosen && CAPUciMove(chosen.UTF8String)) {
                    bestMove = chosen.UTF8String;
                } else {
                    NSLog(@"[ChessAssistantPro] %@ choice unavailable ELO=%d ply=%d lines=%d; using engine bestmove",
                          lowElo ? @"Human" : @"PDF",requestedElo,ply,HELineCount);
                }
                humanLastFen = fen;
                humanLastMove = bestMove;
                humanLastPly = ply;
                humanLastElo = requestedElo;
            }
        } else if (success && stockfish19) {
            // Returning from a high-Elo search must not reuse an old sampled
            // position when the slider changes back to a candidate-law mode.
            humanLastElo=-1;
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

static UISlider *CAPFindEloSlider(UIView *view) {
    if ([view isKindOfClass:UISlider.class]) {
        UISlider *slider = (UISlider *)view;
        if (fabsf(slider.minimumValue) < 0.01f &&
            fabsf(slider.maximumValue - (float)(kEloPresetCount - 1)) < 0.01f)
            return slider;
    }
    for (UIView *child in view.subviews) {
        UISlider *found = CAPFindEloSlider(child);
        if (found) return found;
    }
    return nil;
}

static UILabel *CAPPanelEloLabel(id panel, const char *name) {
    Ivar ivar = class_getInstanceVariable([panel class], name);
    id value = ivar ? object_getIvar(panel, ivar) : nil;
    return [value isKindOfClass:UILabel.class] ? (UILabel *)value : nil;
}

static void CAPSetEloPanelLabels(id panel, int elo) {
    UILabel *value = CAPPanelEloLabel(panel, "_eloValue");
    UILabel *tier = CAPPanelEloLabel(panel, "_eloTier");
    value.text = [NSString stringWithFormat:@"%d", elo];
    tier.text = elo < 1320 ? @"Human-like" :
        (HEPdfBand(elo) ? @"PDF move law" :
         (elo >= 3190 ? @"Maximum" : @"Stockfish 19"));
}

static BOOL CAPImageContainsRange(const struct mach_header_64 *header,
                                  uint64_t address, size_t length, BOOL writable) {
    if (!header || header->magic != MH_MAGIC_64) return NO;
    const uint8_t *command = (const uint8_t *)(header + 1);
    const uint8_t *end = command + header->sizeofcmds;
    for (uint32_t index = 0; index < header->ncmds; ++index) {
        if (command + sizeof(struct load_command) > end) return NO;
        const struct load_command *load = (const struct load_command *)command;
        if (load->cmdsize < sizeof(struct load_command) || command + load->cmdsize > end)
            return NO;
        if (load->cmd == LC_SEGMENT_64 && load->cmdsize >= sizeof(struct segment_command_64)) {
            const struct segment_command_64 *segment = (const struct segment_command_64 *)load;
            if (address >= segment->vmaddr && address + length <= segment->vmaddr + segment->vmsize &&
                (!writable || (segment->initprot & VM_PROT_WRITE))) return YES;
        }
        command += load->cmdsize;
    }
    return NO;
}

static void CAPSyncOriginalEloGlobal(int elo) {
    // Chess.dylib 4.10.22 keeps its active Elo in writable __DATA. Verify the
    // original 14-entry table before touching this version-specific address.
    for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const char *image = _dyld_get_image_name(i);
        if (!image || !strstr(image, "/Chess.dylib")) continue;
        const struct mach_header_64 *header =
            (const struct mach_header_64 *)_dyld_get_image_header(i);
        if (!CAPImageContainsRange(header, 0x6f968, 14 * sizeof(int64_t), NO) ||
            !CAPImageContainsRange(header, 0x26c0ed8, sizeof(int64_t), YES)) return;
        intptr_t slide = _dyld_get_image_vmaddr_slide(i);
        const int64_t *original = (const int64_t *)(slide + 0x6f968);
        if (original[0] != 400 || original[6] != 1600 || original[13] != 3500) {
            NSLog(@"[ChessAssistantPro] Original Elo table changed; skipping global sync");
            return;
        }
        int64_t *current = (int64_t *)(slide + 0x26c0ed8);
        __atomic_store_n(current, (int64_t)elo, __ATOMIC_RELAXED);
        return;
    }
}

static void CAPUpdateStockfishSkillUI(id panel, int elo) {
    UILabel *label = objc_getAssociatedObject(panel, &kStockfishSkillLabelKey);
    if (!label) return;
    if (elo >= 3190) {
        label.text = @"Max · Skill 20/20";
        label.accessibilityLabel = @"Stockfish 19 full strength, skill level 20 of 20, no fixed Elo rating";
    } else if (HEPdfBand(elo)) {
        label.text = [NSString stringWithFormat:@"%d · PDF move law",elo];
        label.accessibilityLabel = [NSString stringWithFormat:
            @"Stockfish 19 candidate move law at preset %d; rating not calibrated, search skill 20 of 20",elo];
    } else if (elo >= 1320) {
        label.text = [NSString stringWithFormat:@"%d · UCI Elo", elo];
        label.accessibilityLabel = [NSString stringWithFormat:
            @"Stockfish 19 UCI ELO limit %d, search skill level 20 of 20",
            elo];
    } else {
        NSDictionary *profile = HEDescribeElo(elo);
        label.text = [NSString stringWithFormat:@"%d · Human mix · Depth %@",
                      elo, profile[@"depth"] ?: @"?"];
        label.accessibilityLabel = [NSString stringWithFormat:
            @"Stockfish 19 uncalibrated human move profile %d, depth %@, up to %@ scored moves; engine search skill 20 of 20",
            elo, profile[@"depth"] ?: @"?", profile[@"multipv"] ?: @"?"];
    }
}

static void CAPConfigureEloSlider(id panel) {
    if (![panel isKindOfClass:UIView.class]) return;
    UISlider *slider = CAPFindEloSlider((UIView *)panel);
    if (!slider) return;
    slider.minimumValue = 0;
    slider.maximumValue = (float)(kEloPresetCount - 1);
    int elo = CAPRequestedStockfishElo();
    [slider setValue:(float)CAPEloPresetIndexFor(elo) animated:NO];
    CAPSyncOriginalEloGlobal(elo);
    CAPSetEloPanelLabels(panel, elo);
    CAPUpdateStockfishSkillUI(panel, elo);
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
    CAPUpdateStockfishSkillUI(panel, CAPRequestedStockfishElo());
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
    CAPConfigureEloSlider(self);
    CAPInsertGodEngine(self);
    CAPInsertStockfishSkill(self);
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
}
- (void)populate {
    %orig;
    CAPConfigureEloSlider(self);
    CAPInsertGodEngine(self);
    CAPInsertStockfishSkill(self);
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
}
- (void)eloSlide:(UISlider *)sender {
    %orig;
    int index = MAX(0, MIN(kEloPresetCount - 1, (int)lroundf(sender.value)));
    [sender setValue:(float)index animated:NO];
    int elo = kEloPresets[index];
    CAPSetEloPanelLabels(self, elo);
    CAPUpdateStockfishSkillUI(self, elo);
}
- (void)eloDone:(UISlider *)sender {
    int index = MAX(0, MIN(kEloPresetCount - 1, (int)lroundf(sender.value)));
    [sender setValue:(float)index animated:NO];
    int elo = kEloPresets[index];
    [[NSUserDefaults standardUserDefaults] setInteger:elo forKey:kStrengthEloKey];
    CAPSyncOriginalEloGlobal(elo);
    CAPSetEloPanelLabels(self, elo);
    CAPUpdateStockfishSkillUI(self, elo);
    engineGeneration.fetch_add(1);
    ++autoPlayGeneration;
    lastScheduledFEN = nil;
    lastPlayedFEN = nil;
    cachedGodMove = cachedGodFEN = nil;
    cachedGodChoice = -1;
    UIView *board = CAPVisibleSwiftBoard();
    if (board) { [board setNeedsLayout]; [board layoutIfNeeded]; }
    NSLog(@"[ChessAssistantPro] ELO preset %d selected (slider stop %d)", elo, index);
}
- (void)creditsTap {
    UIViewController *presenter = ((UIView *)self).window.rootViewController;
    while (presenter.presentedViewController) presenter = presenter.presentedViewController;
    if (!presenter) return;
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:nil
        message:@"Made by JerichBrylle :)" preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK"
        style:UIAlertActionStyleDefault handler:nil]];
    [presenter presentViewController:alert animated:YES completion:nil];
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
