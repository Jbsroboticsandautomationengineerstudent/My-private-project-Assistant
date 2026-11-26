# Chess Assistant IPA triage

## Extraction and inventory

- Source: `Chess_Assistant_4.10.22_yijvlhjo.ipa` (231,682,716 bytes).
- Extracted to `extracted_ipa/`.
- `Payload/Chess.app` has 910 immediate children. The full top-level list is in `app_top_level_inventory.csv`.
- Main executable: `Payload/Chess.app/Chess` (105,249,776 bytes; ARM64 Mach-O executable).
- Only `.dylib` found within the app: `Payload/Chess.app/Frameworks/Chess.dylib` (41,226,784 bytes; ARM64 Mach-O dynamic library).
- `CydiaSubstrate.framework/CydiaSubstrate` is present (379,968 bytes). No ElleKit framework was found.
- Main executable has an `LC_LOAD_WEAK_DYLIB` entry for `@rpath/Chess.dylib`. That library loads `@rpath/CydiaSubstrate.framework/CydiaSubstrate` and imports `_MSHookFunction` and `_MSHookMessageEx`. This is direct evidence of the injection path and runtime hooking.

## Key `Chess.dylib` findings

File offsets below are from `chess_dylib_strings.txt`, not runtime addresses.

- `0x0006cb37`: `loaded v2.3 (SF16 + Maia3)`.
- `0x0006b87b`: `Stockfish `; `0x0006aecb`: `UCI_Elo`; `0x0006aedb`: `bestmove `; `0x0006c526`: `position fen `.
- `0x0006ca09`: `Maia (human-like)`; `0x0006ce05`: `ChessAssist_UseMaia`; `0x0006d7cd`: `maia elo%ld: %@`.
- `0x0006cb52`: jailbreak support path for `maia3_5m.mlpackage`; `0x0006cbc5`: bundled model name. The app contains that Core ML package with a 10,339,456-byte weights file.
- Exported symbols include `_EngineStart`, `_EngineGo`, `_MaiaLoad`, `_MaiaAvailable`, `_MaiaGo`, `_StockfishFenLegal`, and `_StockfishLegalMoves`.
- `0x0006cc46`: `engine poller started`; `0x0006d81e`: `fetch d%ld: %@`; `0x0006d7dd`: `stale resp`; `0x0006d82d`: `no bestmove`.
- `0x0006ce9b` onward names board classes `CHBoardView`, `CHSoloBoardView`, `CHSimpleBoardView`, `CHDiagramBoardView`, and `CHAnalysisBoardView`. `0x0006cef3` logs a hooked `layoutSubviews` call.
- `0x0006cfd0` onward lists Swift `getCurrentFen` mangled symbols. Other strings include `setEncodedMoves:`, `getMainLineTcnString`, `currentFEN`, `fenString`, and `boardFEN`.
- Objective-C class names include `CHAssistBtnHandler`, `CHPad`, and `CHSettingsPanel`. Selectors include `floatBtnTapped`, `handlePan:`, `swMaia:`, `styleSeg:`, `evalSeg:`, and `layoutSubviews`.
- UI evidence includes `UIWindow`, imported `_UIWindowLevelAlert`, imported `CAShapeLayer`, `addSublayer:`, `bezierPath`, `moveToPoint:`, `addLineToPoint:`, `setStrokeColor:`, `setLineWidth:`, and `arrows: %lu`.
- Core ML selectors include `compileModelAtURL:error:`, `modelWithContentsOfURL:configuration:error:`, `predictionFromFeatures:error:`, and `multiArrayValue`.

## Architecture assessment

`Chess.dylib` is the injected tweak. It uses Cydia Substrate to hook host board and game methods, obtains or reconstructs the current FEN, and runs an engine poller. It contains a local Stockfish implementation and a bundled Maia 3 Core ML model. The settings panel offers a Maia switch, strength control, analysis labels, and arrow controls. A floating window and board-associated `CAShapeLayer` drawing appear to render the controls and move arrows.

The tweak's strings provide no evidence of a remote move API. Its visible URLs are Stockfish documentation and NNUE download references. This supports local move calculation; static strings alone cannot rule out dynamically constructed network requests.

## Maia ONNX integration seam

The likely engine boundary is the native C interface exported as `_EngineGo` and `_MaiaGo`, with `_EngineStart`, `_MaiaLoad`, and `_MaiaAvailable` handling initialization and availability. A replacement should preserve the upstream FEN capture and downstream move/result format, then route position evaluation at this boundary to the ONNX runtime. The package already has a Maia Core ML path, so first compare its expected inputs and outputs with the ONNX model before patching code. Exact function signatures, callers, and struct layouts require disassembly or runtime tracing; the string pass does not establish them.
