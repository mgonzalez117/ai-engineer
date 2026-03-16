import {
  AfterViewInit,
  Component,
  ElementRef,
  EventEmitter,
  Output,
  ViewChild,
} from '@angular/core';
import { Chess, Square } from 'chess.js';

import '@lichess-org/chessground/assets/chessground.base.css';

import { Chessground } from '@lichess-org/chessground';
import type { Api } from '@lichess-org/chessground/api';
import type { Key } from '@lichess-org/chessground/types';

type PiecesTheme = 'tournament' | 'metal' | 'trimmed';
type BoardTheme = 'maple' | 'marble' | 'metalboard';

@Component({
  selector: 'app-chessboard',
  standalone: true,
  template: `
    <div class="cg-controls">
      <div class="cg-control">
        <label for="pieces">Pièces</label>
        <select id="pieces" (change)="changePieces($event)">
          <option value="tournament" selected>Tournament</option>
          <option value="metal">Metal</option>
          <option value="trimmed">Trimmed</option>
        </select>
      </div>

      <div class="cg-control">
        <label for="board">Board</label>
        <select id="board" (change)="changeBoard($event)">
          <option value="maple" selected>Maple</option>
          <option value="marble">Marble</option>
          <option value="metalboard">Metal</option>
        </select>
      </div>
    </div>

    <div #board class="cg-board-host"></div>
  `,
})
export class ChessboardComponent implements AfterViewInit {
  @ViewChild('board', { static: true }) el!: ElementRef<HTMLDivElement>;

  @Output() fenChange = new EventEmitter<string>();
  @Output() dragFilterChange = new EventEmitter<Key | null>();

  private game = new Chess();
  private ground!: Api;

  private currentPieces: PiecesTheme = 'tournament';
  private currentBoard: BoardTheme = 'maple';

  /**
   * Case source actuellement "en cours d'interaction"
   * Sert uniquement au filtre front des recommandations.
   */
  private activeFrom: Key | null = null;

  ngAfterViewInit(): void {
    this.el.nativeElement.classList.add(this.currentPieces, this.currentBoard);

    this.ground = Chessground(this.el.nativeElement, {
      selectable: { enabled: true },
      draggable: { enabled: true, showGhost: true },
      movable: {
        free: false,
        showDests: true,
        color: 'both',
        dests: this.computeDests(),
        events: {
          after: (from, to) => this.onMove(from as Key, to as Key),
        },
      },
      events: {
        select: (key) => this.onSelect(key as Key),
      },
      highlight: { lastMove: true, check: true },
    });

    this.sync();

    queueMicrotask(() => {
      this.fenChange.emit(this.game.fen());
    });

    /**
     * Fin d'interaction sans coup joué :
     * on retire simplement le filtre local.
     * Pas de nouvelle recherche.
     */
    this.el.nativeElement.addEventListener('pointerup', this.onPointerUp);
    this.el.nativeElement.addEventListener('touchend', this.onPointerUp);
  }

  changePieces(event: Event) {
    const value = (event.target as HTMLSelectElement).value as PiecesTheme;
    this.el.nativeElement.classList.remove(this.currentPieces);
    this.el.nativeElement.classList.add(value);
    this.currentPieces = value;
  }

  changeBoard(event: Event) {
    const value = (event.target as HTMLSelectElement).value as BoardTheme;
    this.el.nativeElement.classList.remove(this.currentBoard);
    this.el.nativeElement.classList.add(value);
    this.currentBoard = value;
  }

 private onSelect(key: Key) {
  const piece = this.game.get(key as Square);

  if (!piece) {
    return;
  }

  const turn = this.game.turn();
  const pieceColor = piece.color;

  if (turn !== pieceColor) {
    return;
  }

  this.activeFrom = key;
  this.dragFilterChange.emit(key);
}

  private onMove(from: Key, to: Key) {
    const move = this.game.move({ from, to, promotion: 'q' });

    if (!move) {
      this.sync();
      this.clearDragFilter();
      return;
    }

    this.sync();
    this.ground.set({
      movable: { dests: this.computeDests() },
    });

    // Le coup est joué : on enlève le filtre temporaire
    this.clearDragFilter();

    // Puis on laisse le reste de l'app faire sa logique habituelle
    this.fenChange.emit(this.game.fen());
  }

  private onPointerUp = () => {
    /**
     * Si on relâche sans qu'un coup légal ait été joué,
     * on revient à l'état initial côté filtre.
     */
    this.clearDragFilter();
  };

  private clearDragFilter() {
    if (this.activeFrom !== null) {
      this.activeFrom = null;
      this.dragFilterChange.emit(null);
    }
  }

  private sync() {
    this.ground.set({ fen: this.game.fen() });
  }

  private computeDests(): Map<Key, Key[]> {
    const dests = new Map<Key, Key[]>();
    const moves = this.game.moves({ verbose: true }) as Array<{ from: Key; to: Key }>;

    for (const m of moves) {
      const arr = dests.get(m.from);
      if (arr) arr.push(m.to);
      else dests.set(m.from, [m.to]);
    }

    return dests;
  }
}
