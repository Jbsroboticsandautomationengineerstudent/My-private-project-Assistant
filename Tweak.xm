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
#include <string>

extern char **environ;

static NSString *const kModelPath = @"/var/jb/Library/Application Support/ChessAssistantPro/maia3_79m_int32.onnx";
static NSString *const kAutoPlayKey = @"ChessAssistantPro_AutoPlay";
static NSString *const kAutoPlayMinDelayKey = @"ChessAssistantPro_AutoPlayMinDelay";
static NSString *const kAutoPlayMaxDelayKey = @"ChessAssistantPro_AutoPlayMaxDelay";
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
    if ([value respondsToSelector:@selector(lanMove)]) {
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
    if (!view || depth > 4 || *count >= 2000) return;
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

typedef void (*CAPLayoutIMP)(UIView *, SEL);
static CAPLayoutIMP originalSwiftBoardLayout = nullptr;
static BOOL capBoardProbeQueued = NO; // UIKit layout runs on the main thread.
static void CAPSwiftBoardLayout(UIView *board, SEL selector) {
    originalSwiftBoardLayout(board, selector);
    if (!board.window || capBoardProbeQueued) return;
    capBoardProbeQueued = YES;
    dispatch_async(dispatch_get_main_queue(), ^{
        capBoardProbeQueued = NO;
        CAPVisibleSwiftBoard();
    });
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
            NSLog(@"[ChessAssistantPro] Auto-Play [direct] %@; no verified move API for %@",
                  CAPVisibleSwiftBoard() ? @"Swift board detected" : @"matching board not found", uci);
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

static BOOL CAPExternalEngineMove(const char *path, const char *fen,
                                  BOOL stockfish19, std::string &bestMove) {
    if (!path || !fen || access(path, X_OK) != 0) return NO;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    // Accessed only from the serial God Engine queue. Keep Stockfish warm so
    // later positions do not pay process launch and NNUE initialization costs.
    static int cachedInput = -1, cachedOutput = -1;
    static pid_t cachedPid = 0;
    if (stockfish19 && cachedPid > 0) {
        int status = 0;
        pid_t exited;
        do { exited = waitpid(cachedPid, &status, WNOHANG); }
        while (exited < 0 && errno == EINTR);
        if (exited != 0) {
            close(cachedInput); close(cachedOutput);
            cachedInput = cachedOutput = -1;
            cachedPid = 0;
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
    CFAbsoluteTime deadline = CFAbsoluteTimeGetCurrent() + 5.0;
    if (!reused) {
        if (!CAPWrite(input[1], "uci\n") || !CAPWaitForLine(output[0], "uciok", deadline)) goto done;
        if (stockfish19 &&
            (!CAPWrite(input[1], "setoption name Threads value 2\n") ||
             !CAPWrite(input[1], "setoption name Hash value 32\n") ||
             !CAPWrite(input[1], "setoption name MultiPV value 1\n") ||
             !CAPWrite(input[1], "setoption name Skill Level value 20\n") ||
             !CAPWrite(input[1], "setoption name UCI_LimitStrength value false\n"))) goto done;
        if (!CAPWrite(input[1], "isready\n") || !CAPWaitForLine(output[0], "readyok", deadline)) goto done;
    }
    if (!CAPWrite(input[1], std::string("position fen ") + fen + "\n")) goto done;
    if (!CAPWrite(input[1], stockfish19 ? "go movetime 200\n" : "go depth 16\n")) goto done;
    deadline = CFAbsoluteTimeGetCurrent() + (stockfish19 ? 3.0 : 15.0);
    while (CAPReadLine(output[0], line, deadline)) {
        if (line.find("CRITICAL ERROR") != std::string::npos) break;
        if (line.compare(0, 9, "bestmove ") != 0) continue;
        size_t end = line.find(' ', 9);
        bestMove = line.substr(9, end == std::string::npos ? end : end - 9);
        success = CAPUciMove(bestMove);
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
                found = CAPExternalEngineMove(kTriumviratusPath, position.UTF8String, NO, move);
            if (!found)
                found = CAPExternalEngineMove(kStockfish19Path, position.UTF8String, YES, move);
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
static char kGodEngineRowKey;

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
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
}
- (void)populate {
    %orig;
    CAPInsertGodEngine(self);
    CAPInsertAutoPlay(self);
    CAPInstallDumpGesture(self);
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
        CAPDiscLog(@"diagnostics installed SwiftBoard=%p WebSocket=%p", swiftBoard, webSocketTask);
        NSLog(@"[ChessAssistantPro] Maia=%p Engine=%p", maia, engine);
    }
}
