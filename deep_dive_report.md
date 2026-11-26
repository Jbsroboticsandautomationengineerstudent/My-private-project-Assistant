# Chess.dylib Objective-C and engine deep dive

The addresses below are unslid Mach-O virtual addresses. The Objective-C declarations are reconstructed from selector names and type encodings; unnamed argument labels are ours. `id` reflects the encoding's generic object type.

## Complete Objective-C class method lists

```objc
@interface CHAssistBtnHandler
+ (void)floatBtnTapped;                     // v16@0:8; IMP 0x4e028
+ (void)handlePan:(id)sender;               // v24@0:8@16; IMP 0x4e03c
@end

@interface CHPad
- (void)drawTextInRect:(CGRect)rect;         // v48@0:8{CGRect={CGPoint=dd}{CGSize=dd}}16; IMP 0x4e13c
- (CGSize)intrinsicContentSize;              // {CGSize=dd}16@0:8; IMP 0x4e190
@end

@interface CHSettingsPanel
- (id)initWithFrame:(CGRect)frame;           // @48@0:8{CGRect={CGPoint=dd}{CGSize=dd}}16; IMP 0x4e6e4
- (void)bgTap:(id)sender;                    // v24@0:8@16; IMP 0x4e7dc
- (id)lbl:(id)text size:(double)size weight:(double)weight color:(id)color; // @48@0:8@16d24d32@40; IMP 0x4e864
- (id)sep;                                  // @16@0:8; IMP 0x4e964
- (id)group:(id)arg;                        // @24@0:8@16; IMP 0x4ea4c
- (id)rowTitle:(id)title control:(id)control; // @32@0:8@16@24; IMP 0x4ee20
- (void)styleSeg:(id)sender;                // v24@0:8@16; IMP 0x4f0a4
- (id)switchOn:(BOOL)on sel:(SEL)selector;   // @28@0:8B16:20; IMP 0x4f2e4
- (id)chip:(id)arg color:(id)color;         // @32@0:8@16@24; IMP 0x4f3ac
- (id)bigBtn:(id)arg color:(id)color sel:(SEL)selector; // @40@0:8@16@24:32; IMP 0x4f524
- (id)smallBtn:(id)arg sel:(SEL)selector;   // @32@0:8@16:24; IMP 0x4f70c
- (id)statsCard;                            // @16@0:8; IMP 0x4f910
- (void)build;                              // v16@0:8; IMP 0x506e4
- (void)populate;                           // v16@0:8; IMP 0x51568
- (void)eloSlide:(id)sender;                // v24@0:8@16; IMP 0x52acc
- (void)eloDone:(id)sender;                 // v24@0:8@16; IMP 0x52c00
- (void)evalSeg:(id)sender;                 // v24@0:8@16; IMP 0x53058
- (void)arrSeg:(id)sender;                  // v24@0:8@16; IMP 0x530c4
- (void)thickSeg:(id)sender;                // v24@0:8@16; IMP 0x53128
- (void)opSlide:(id)sender;                 // v24@0:8@16; IMP 0x531b8
- (void)swAnalysis:(id)sender;              // v24@0:8@16; IMP 0x53220
- (void)swLabels:(id)sender;                // v24@0:8@16; IMP 0x532ec
- (void)swColor:(id)sender;                 // v24@0:8@16; IMP 0x53350
- (void)swMaia:(id)sender;                  // v24@0:8@16; IMP 0x533b4
- (void)enableTap;                          // v16@0:8; IMP 0x53414
- (void)debugTap;                           // v16@0:8; IMP 0x53588
- (void)creditsTap;                         // v16@0:8; IMP 0x537a0
- (void)closeTap;                           // v16@0:8; IMP 0x53894
- (void)animateIn;                          // v16@0:8; IMP 0x53898
- (void)animateOut;                         // v16@0:8; IMP 0x53a58
// -[CHSettingsPanel .cxx_destruct]          // v16@0:8; IMP 0x53c74 (compiler-generated)
+ (void)show;                               // v16@0:8; IMP 0x4e1d4
@end
```

No Objective-C class in this library defines a method named `EngineGo` or `MaiaGo`. They are exported C symbols at `0x5fe10` and `0x63ef0`. Their C parameter types are absent from Objective-C metadata and the stripped symbol table. Static direct calls from unnamed C/C++ routines are listed in `engine_and_hook_calls.txt`.

## FEN capture

The hook initializer at `0x54418` calls `MSHookMessageEx` and `MSHookFunction`. Its string references include `layoutSubviews`, `getCurrentFen`, `getMainLineTcnString`, and `setEncodedMoves:`. It probes Swift mangled symbols including `$s16EngineController19ChessGameControllerC13getCurrentFenSSyF`, with several alternative spellings/modules. The nearby fallback strings name `currentFEN`, `currentFen`, `fenString`, `boardFEN`, `positionFEN`, `initialFEN`, and `getCurrentFen`, plus traversals through `game`, `chessGame`, `boardModel`, `viewModel`, and controller/delegate chains. This shows multiple FEN acquisition paths rather than one guaranteed host method.

## UI injection and drawing

The hook routine at `0x56460` obtains `UIWindow`, uses selector `hitTest:withEvent:`, and calls `MSHookMessageEx`. It then obtains `CHAppDelegate`, uses `applicationDidBecomeActive:`, and calls `MSHookMessageEx` again. The imported `UIWindowLevelAlert`, `makeKeyAndVisible`, `initWithWindowScene:`, and `setWindowLevel:` support creation of a floating window. `CHAssistBtnHandler` handles its button tap and pan. `CHPad` itself only overrides text drawing and intrinsic sizing. Arrows are drawn elsewhere with `CAShapeLayer`, paths, and board layout hooks; no `-[CHPad drawArrow:to:]` method exists.

## Maia Core ML path

The bundled model has these declared MultiArray features:

```text
input   tokens        [1, 64, 96]  Float32
input   self_elo      [1]          Int32
input   oppo_elo      [1]          Int32
output  move_logits   [1, 4352]   Float32
output  value_logits  [1, 3]      Float32
```

`_MaiaLoad` is directly called from the startup routine at `0x54118`. `_MaiaGo` is called at `0x5d3c4`; the call site passes four ARM64 argument registers, including equal values in `x1` and `x2` (consistent with the two Elo inputs) and a stack address in `x3` (consistent with an asynchronous result block). The `_MaiaGo` routine references the five feature names above and uses Core ML model compilation/loading and `predictionFromFeatures:error:` selectors. The library also contains the block type encoding `v104@?0{MaiaResult=[8c]ddBi[4[8c]][4d]}8`, which indicates a void block receiving a `MaiaResult` structure. Field meanings and the exact C declaration remain unverified.

The likely path is FEN -> position validation/tokenization -> `tokens` plus Elo values -> Core ML prediction -> legal-move filtering/ranking of `move_logits` -> `MaiaResult` callback -> overlay update. The code and model metadata support this flow, but the exact token encoding and logit-to-move map require instruction-level tracing or runtime observation. An ONNX replacement should preserve the surrounding FEN and result interfaces after those two mappings are established.
