# Codex Task: Swift board Auto-Play via the board's own touch handlers

## Verified facts (from the real Chess binary, ObjC metadata dump)
- `_TtC15SwiftChessBoard10ChessBoard` has NO move method. Only: rotate:, layoutSubviews, initWith*.
- The board's input class is `_TtC15SwiftChessBoard28PiecesOverlayWithTouchEvents` with exactly these ObjC methods:
  `touchesBegan:withEvent:`  (IMP 0x2767dcc)
  `touchesMoved:withEvent:`  (IMP 0x2768990)
  `touchesEnded:withEvent:`  (IMP 0x2768e44)
  `touchesCancelled:withEvent:` (IMP 0x2768e50)
- Other board classes: `_TtC15SwiftChessBoard9PieceView`, `PieceDragOrMoveView`, `MoveDestinationsView`, `TappableView` (has `tapGestureProcessWithGesture:`), `PromotionPopupViewController`.
- All game logic (move validation, sending to server) is pure Swift with no ObjC selector, so there is no `makeMove`-style API to call. The only reachable entry is the touch overlay.
- Previous HID/IOKit synthetic touches were removed (commit 4962850). Do NOT re-add HID. Instead call the overlay's touch methods directly in-process.

## What to implement in Tweak.xm (replace the dead end in CAPAutoPlay for Swift boards)
1. Find the overlay: walk the subviews of `CAPVisibleSwiftBoard()` and return the first view whose class is `_TtC15SwiftChessBoard28PiecesOverlayWithTouchEvents` (use NSClassFromString + isKindOfClass). If missing, log `[CAP-AP] overlay not found` and dump the subview tree (class + frame, 5 levels).
2. Add a `CAPFakeTouch : UITouch` subclass overriding: `locationInView:` and `previousLocationInView:` (return the stored point converted into the requested view's space via the window), `phase`, `view`, `window`, `tapCount` (1), `timestamp`, `type` (UITouchTypeDirect), `majorRadius`. Store point in window coordinates.
3. Add a `CAPFakeEvent : UIEvent` subclass with `allTouches`/`touchesForView:` returning the fake touch, `timestamp`, `type` UIEventTypeTouches.
4. Square -> point: use the overlay's own bounds (not the board's): square = overlay.bounds.width / 8; file/rank to x/y with orientation flipped when black is at the bottom. Determine orientation by finding any `_TtC15SwiftChessBoard9PieceView` for our king (or compare the screen y of PieceViews on rank 1 vs rank 8). Log the choice.
5. Play a move as drag: touchesBegan at source, 4x touchesMoved interpolated to destination (about 16 ms apart, dispatch_after on main queue), touchesEnded at destination. If the board did not change after 0.6 s, retry once as tap-source then tap-destination (began+ended on source, 150 ms later began+ended on destination).
6. Success check without FEN selectors: after the attempt, confirm that a PieceView now sits at the destination square center (walk `_TtC15SwiftChessBoard9PieceView` frames converted to window coords). Log result `[CAP-AP] ok/fail`.
7. Promotion: if a `PromotionPopupViewController` view appears in the key window within 0.8 s, tap the choice matching the 5th UCI char (default queen) with the same fake-touch tap.
8. Observer logs (needed if it fails): hook `touchesBegan:withEvent:` on the overlay class with MSHookMessageEx and log every REAL touch the user makes (point in overlay coords, phase, touch class). Compare real vs fake touch to fix any mismatch.

## Rules
- Keep CHBoardView path and engine code unchanged.
- Log prefix `[CAP-AP]`. Build must pass `make package`. Report changed line ranges.
- If the board ignores fake touches (likely cause: Swift reads `UITouch` private fields or uses `event.coalescedTouches`), implement the observer from step 8 first and send me the log of one real drag.
