#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <substrate.h>
#import <dlfcn.h>
#import <dispatch/dispatch.h>
#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <string>

extern char **environ;

static NSString *const kModelPath = @"/var/jb/Library/Application Support/ChessAssistantPro/maia3_79m_int32.onnx";
static NSString *const kAutoPlayKey = @"ChessAssistantPro_AutoPlay";
static NSString *const kGodEngineKey = @"godEngineChoice";
static const char *kStockfish19Path = "/var/jb/usr/bin/stockfish19";
static const char *kTriumviratusPath = "/var/jb/usr/bin/triumviratus";

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
static NSString *lastRequestedFEN = nil;
static NSString *lastPlayedFEN = nil;

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

static UIView *CAPFindBoard(UIView *view) {
    Class boardClass = NSClassFromString(@"CHBoardView");
    if (boardClass && [view isKindOfClass:boardClass] && !view.hidden) return view;
    for (UIView *child in view.subviews) {
        UIView *found = CAPFindBoard(child);
        if (found) return found;
    }
    return nil;
}

static UIView *CAPVisibleBoard(void) {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            UIView *found = CAPFindBoard(window);
            if (found) return found;
        }
    }
    return nil;
}

static void CAPAutoPlay(NSString *move, NSString *fen) {
    if (![[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey] ||
        !fen.length || ![lastRequestedFEN isEqualToString:fen] ||
        [lastPlayedFEN isEqualToString:fen]) return;
    NSString *uci = CAPNormalizeMove(move);
    if (uci.length != 4 && uci.length != 5) return;
    int sf = [uci characterAtIndex:0] - 'a', sr = [uci characterAtIndex:1] - '1';
    int df = [uci characterAtIndex:2] - 'a', dr = [uci characterAtIndex:3] - '1';
    if (sf < 0 || sf > 7 || sr < 0 || sr > 7 ||
        df < 0 || df > 7 || dr < 0 || dr > 7) return;
    @try {
        UIView *board = CAPVisibleBoard();
        if (!board || ![board respondsToSelector:@selector(makeMove:throughTouches:)] ||
            ![board respondsToSelector:@selector(chessGame)]) return;
        BOOL whiteToMove = [fen containsString:@" w "];
        SEL permission = whiteToMove ? @selector(allowUserToMakeWhiteMoves)
                                     : @selector(allowUserToMakeBlackMoves);
        if ([board respondsToSelector:permission] &&
            !((BOOL (*)(id, SEL))objc_msgSend)(board, permission)) return;
        id game = ((id (*)(id, SEL))objc_msgSend)(board, @selector(chessGame));
        SEL validMove = @selector(isValidMoveFromSourceRank:SourceFile:DestRank:DestFile:);
        Class moveClass = NSClassFromString(@"CHMoveSourceDestination");
        SEL initializer = @selector(initSR:SF:DR:DF:PP:);
        if (!game || ![game respondsToSelector:validMove] ||
            !moveClass || ![moveClass instancesRespondToSelector:initializer]) return;
        char promotion = uci.length == 5 ? (char)[uci characterAtIndex:4] : 0;
        for (int base = 0; base <= 1; ++base) {
            char sourceRank = (char)(sr + base), sourceFile = (char)(sf + base);
            char destRank = (char)(dr + base), destFile = (char)(df + base);
            id candidate = ((id (*)(id, SEL, char, char, char, char, char))objc_msgSend)
                ([moveClass alloc], initializer, sourceRank, sourceFile, destRank, destFile, promotion);
            if (![candidate respondsToSelector:@selector(lanMove)]) continue;
            NSString *lan = ((id (*)(id, SEL))objc_msgSend)(candidate, @selector(lanMove));
            if (![[CAPNormalizeMove(lan) lowercaseString] isEqualToString:uci]) continue;
            BOOL legal = ((BOOL (*)(id, SEL, char, char, char, char))objc_msgSend)
                (game, validMove, sourceRank, sourceFile, destRank, destFile);
            if (!legal) return;
            ((void (*)(id, SEL, id, BOOL))objc_msgSend)
                (board, @selector(makeMove:throughTouches:), candidate, YES);
            lastPlayedFEN = [fen copy];
            return;
        }
    } @catch (NSException *exception) {
        NSLog(@"[ChessAssistantPro] Auto-Play skipped: %@", exception);
    }
}

static void CAPMaiaGo(const char *fen, uintptr_t selfElo, uintptr_t oppoElo,
                      MaiaCallback callback) {
    if (!originalMaiaGo) return;
    size_t length = fen ? strnlen(fen, 256) : 0;
    NSString *position = length > 0 && length < 256 ?
        [[NSString alloc] initWithBytes:fen length:length encoding:NSUTF8StringEncoding] : nil;
    if (position) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (![lastRequestedFEN isEqualToString:position]) lastPlayedFEN = nil;
            lastRequestedFEN = [position copy];
        });
    }
    MaiaCallback forwarded = ^(MaiaResult result) {
        if (position && [[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey]) {
            size_t moveLength = strnlen(result.move, sizeof(result.move));
            if (moveLength >= 4 && moveLength <= 5) {
                NSString *move = [[NSString alloc] initWithBytes:result.move length:moveLength
                                                        encoding:NSASCIIStringEncoding];
                dispatch_async(dispatch_get_main_queue(), ^{ CAPAutoPlay(move, position); });
            }
        }
        if (callback) callback(result);
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
    if (length == 0 || length >= 256 || strpbrk(fen, "\r\n")) {
        originalEngineGo(fen, arg1, arg2, arg3, callback);
        return;
    }
    BOOL useTriumviratus = [[NSUserDefaults standardUserDefaults] integerForKey:kGodEngineKey] == 1;
    if (access(useTriumviratus ? kTriumviratusPath : kStockfish19Path, X_OK) != 0 &&
        access(kStockfish19Path, X_OK) != 0) {
        originalEngineGo(fen, arg1, arg2, arg3, callback);
        return;
    }
    NSString *position = [[NSString alloc] initWithBytes:fen length:length encoding:NSUTF8StringEncoding];
    if (!position) { originalEngineGo(fen, arg1, arg2, arg3, callback); return; }
    EngineCallback copied = [callback copy];
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
                originalEngineGo(position.UTF8String, arg1, arg2, arg3, copied);
                return;
            }
            EngineLine result = {};
            strlcpy(result.move, move.c_str(), sizeof(result.move));
            if (copied) copied(&result, 1);
        });
    });
}

static UIButton *CAPFindPauseButton(UIView *view) {
    if ([view isKindOfClass:UIButton.class]) {
        UIButton *button = (UIButton *)view;
        NSString *title = [button titleForState:UIControlStateNormal] ?: button.titleLabel.text;
        if ([title rangeOfString:@"Pause Assistant" options:NSCaseInsensitiveSearch].location != NSNotFound)
            return button;
    }
    for (UIView *child in view.subviews) {
        UIButton *button = CAPFindPauseButton(child);
        if (button) return button;
    }
    return nil;
}

static char kAutoPlayRowKey;
static void CAPInsertAutoPlay(id panel) {
    if (![panel isKindOfClass:UIView.class]) return;
    UIView *existing = objc_getAssociatedObject(panel, &kAutoPlayRowKey);
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
    if (!stack || ![panel respondsToSelector:@selector(switchOn:sel:)] ||
        ![panel respondsToSelector:@selector(rowTitle:control:)]) return;
    UISwitch *toggle = ((id (*)(id, SEL, BOOL, SEL))objc_msgSend)
        (panel, @selector(switchOn:sel:),
         [[NSUserDefaults standardUserDefaults] boolForKey:kAutoPlayKey],
         @selector(cap_autoPlayChanged:));
    UIView *row = ((id (*)(id, SEL, id, id))objc_msgSend)
        (panel, @selector(rowTitle:control:), @"Auto-Play", toggle);
    if (![row isKindOfClass:UIView.class]) return;
    NSUInteger index = [stack.arrangedSubviews indexOfObject:child];
    if (index == NSNotFound) return;
    [stack insertArrangedSubview:row atIndex:index];
    objc_setAssociatedObject(panel, &kAutoPlayRowKey, row, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static char kGodEngineRowKey;
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

%hook CHSettingsPanel
- (void)populate {
    %orig;
    CAPInsertGodEngine(self);
    CAPInsertAutoPlay(self);
}
%new
- (void)cap_godEngineChanged:(UISegmentedControl *)sender {
    [[NSUserDefaults standardUserDefaults] setInteger:(sender.selectedSegmentIndex == 1 ? 1 : 0)
                                               forKey:kGodEngineKey];
}
%new
- (void)cap_autoPlayChanged:(UISwitch *)sender {
    [[NSUserDefaults standardUserDefaults] setBool:sender.on forKey:kAutoPlayKey];
}
%end

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
        NSLog(@"[ChessAssistantPro] Maia=%p Engine=%p", maia, engine);
    }
}
