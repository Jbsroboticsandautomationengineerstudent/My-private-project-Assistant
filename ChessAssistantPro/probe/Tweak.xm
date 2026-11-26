#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <substrate.h>
#import <dlfcn.h>
#import <mach-o/loader.h>
#import <mach-o/nlist.h>
#import <mach-o/dyld.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <string>
#include <math.h>

// Probe-only dylib for a sideloaded (non-jailbroken) Chess app.
// Read-only: logs the Swift board structure and real touches. No Auto-Play.

static BOOL CAPMoveWord(NSString *name) {
    return [name rangeOfString:@"move" options:NSCaseInsensitiveSearch].location != NSNotFound &&
           [name rangeOfString:@"remove" options:NSCaseInsensitiveSearch].location == NSNotFound;
}

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
        if ([responder isKindOfClass:UIViewController.class]) break;
        responder = responder.nextResponder;
    }
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

// ---- Board dump trigger + share button ----
static UIWindow *CAPKeyWindow(void) {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) continue;
        for (UIWindow *w in ((UIWindowScene *)scene).windows) if (w.isKeyWindow) return w;
    }
    return nil;
}

static void CAPShareDump(void) {
    NSString *path = CAPDumpPath();
    if (!path || ![[NSFileManager defaultManager] fileExistsAtPath:path]) return;
    UIViewController *top = CAPKeyWindow().rootViewController;
    while (top.presentedViewController) top = top.presentedViewController;
    if (!top) return;
    UIActivityViewController *vc = [[UIActivityViewController alloc]
        initWithActivityItems:@[[NSURL fileURLWithPath:path]] applicationActivities:nil];
    vc.popoverPresentationController.sourceView = top.view;
    vc.popoverPresentationController.sourceRect = CGRectMake(CGRectGetMidX(top.view.bounds), 80, 1, 1);
    [top presentViewController:vc animated:YES completion:nil];
}

@interface CAPShareHelper : NSObject
+ (void)tapped;
@end
@implementation CAPShareHelper
+ (void)tapped { CAPShareDump(); }
@end

static void CAPAddShareButton(UIWindow *window) {
    static BOOL added = NO;
    if (added || !window) return;
    added = YES;
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    [button setTitle:@"Share dump" forState:UIControlStateNormal];
    button.backgroundColor = [UIColor colorWithWhite:0 alpha:0.7];
    button.layer.cornerRadius = 8;
    [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    button.frame = CGRectMake(8, 60, 110, 32);
    [button addTarget:CAPShareHelper.class action:@selector(tapped)
     forControlEvents:UIControlEventTouchUpInside];
    [window addSubview:button];
    CAPDiscLog(@"share button added");
}

static void CAPDumpSwiftBoard(UIView *board) {
    Class cls = [board class];
    Dl_info info = {};
    BOOL located = dladdr((__bridge const void *)cls, &info) != 0;
    const char *image = located && info.dli_fname ? info.dli_fname : class_getImageName(cls);
    const struct mach_header_64 *header = located ? (const struct mach_header_64 *)info.dli_fbase : nullptr;
    CAPDiscLog(@"===== SWIFT BOARD DUMP board=%p image=%s file=%@ =====", board, image ?: "?", CAPDumpPath());
    for (Class c = cls; c; c = class_getSuperclass(c)) {
        CAPDumpClass(c);
        if (c == NSObject.class) break;
    }
    if (image) {
        unsigned n = 0;
        const char **names = objc_copyClassNamesForImage(image, &n);
        CAPDiscLog(@"MODULE image=%s classes=%u", image, n);
        for (unsigned i = 0; i < n; ++i)
            if (strncmp(names[i], "_TtC15SwiftChessBoard", 21) == 0) CAPDumpClass(objc_getClass(names[i]));
        free(names);
    }
    CAPDumpSwiftSymbols(header);
    NSUInteger count = 0;
    CAPDumpSubviewTree(board, 0, &count);
    CAPDiscLog(@"VIEWS logged=%lu", (unsigned long)count);
    CAPDumpResponderChain(board);
    CAPDiscLog(@"===== END SWIFT BOARD DUMP =====");
}

typedef void (*CAPLayoutIMP)(UIView *, SEL);
static CAPLayoutIMP originalBoardLayout = nullptr;
static __weak UIView *capDumpedBoard = nil;
static void CAPBoardLayout(UIView *board, SEL selector) {
    originalBoardLayout(board, selector);
    if (!board.window || capDumpedBoard == board) return;
    capDumpedBoard = board;
    __weak UIView *weak = board;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 1500 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
        UIView *b = weak;
        if (!b || !b.window) return;
        @try { CAPDumpSwiftBoard(b); } @catch (NSException *e) { CAPDiscLog(@"DUMP failed %@", e); }
        CAPAddShareButton(b.window);
    });
}

__attribute__((constructor)) static void CAPProbeInit(void) {
    @autoreleasepool {
        CAPDiscLog(@"probe loaded, dump file %@", CAPDumpPath());
        Class board = NSClassFromString(@"_TtC15SwiftChessBoard10ChessBoard");
        if (board) {
            Method m = class_getInstanceMethod(board, @selector(layoutSubviews));
            originalBoardLayout = m ? (CAPLayoutIMP)method_getImplementation(m) : nullptr;
            if (originalBoardLayout)
                MSHookMessageEx(board, @selector(layoutSubviews), (IMP)CAPBoardLayout, (IMP *)&originalBoardLayout);
        }
        CAPInstallTouchObservers();
        CAPDiscLog(@"probe installed board=%p", board);
    }
}
