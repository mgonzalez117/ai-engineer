import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { AgentResponse, AgentRecommendation } from '../service/api/agent.types';

@Component({
  selector: 'app-agent-sidebar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <aside class="sidebar">
      <div class="sidebar-header">
        <div>
          <p class="eyebrow">Assistant d'analyse</p>
          <h2>Recommandations</h2>
        </div>

        <span class="source-badge" *ngIf="suggestions?.source">
          {{ suggestions?.source }}
        </span>
      </div>

      <section class="card">
        <p class="label">Position actuelle</p>
        <p class="fen">{{ fen || 'Aucune position disponible' }}</p>
      </section>

      <section class="card" *ngIf="loading">
        <p class="status">Chargement des recommandations...</p>
      </section>

      <section class="card error" *ngIf="error">
        <p class="status">{{ error }}</p>
      </section>

      <ng-container *ngIf="!loading && !error && suggestions as data">
        <section class="card" *ngIf="data.evaluation as eval">
          <p class="label">Évaluation</p>
          <p class="eval">
            <ng-container *ngIf="eval.type === 'cp'">
              {{ formatCentipawns(eval.value) }}
            </ng-container>
            <ng-container *ngIf="eval.type === 'mate'">
              Mat en {{ eval.value }}
            </ng-container>
          </p>
        </section>

        <section class="card" *ngIf="data.recommendations?.length">
          <div class="section-head">
            <h3>Coups suggérés</h3>
            <span class="count">{{ data.recommendations?.length }}</span>
          </div>

          <div class="moves-list">
            <article
              class="move-card"
              *ngFor="let rec of data.recommendations; let i = index"
            >
              <div class="move-rank">#{{ i + 1 }}</div>

              <div class="move-main">
                <div class="move-top">
                  <strong class="move-san">{{ rec.san || rec.uci }}</strong>
                  <span class="move-uci" *ngIf="rec.san">{{ rec.uci }}</span>
                </div>

                <p class="opening" *ngIf="rec.opening?.name">
                  {{ rec.opening?.eco ? rec.opening?.eco + ' · ' : '' }}{{ rec.opening?.name }}
                </p>

                <div class="stats" *ngIf="hasStats(rec)">
                  <span *ngIf="rec.averageRating">Elo moyen {{ rec.averageRating }}</span>
                  <span *ngIf="rec.white !== undefined">Blancs {{ rec.white }}</span>
                  <span *ngIf="rec.draws !== undefined">Nulles {{ rec.draws }}</span>
                  <span *ngIf="rec.black !== undefined">Noirs {{ rec.black }}</span>
                </div>
              </div>
            </article>
          </div>
        </section>

        <section class="card" *ngIf="data.context?.length">
          <div class="section-head">
            <h3>Contexte</h3>
            <span class="count">{{ data.context?.length }}</span>
          </div>

          <div class="context-list">
            <article class="context-item" *ngFor="let item of data.context | slice:0:3">
              <p class="context-score" *ngIf="item.score !== undefined">
                Pertinence {{ item.score | number:'1.2-2' }}
              </p>
              <p class="context-text">{{ item.text }}</p>
            </article>
          </div>
        </section>

        <section class="card empty" *ngIf="!data.recommendations?.length && !data.context?.length">
          <p class="status">Aucune recommandation disponible pour cette position.</p>
        </section>
      </ng-container>
    </aside>
  `,
  styleUrl: './agent-sidebar.css',
})
export class AgentSidebarComponent {
  @Input() fen = '';
  @Input() loading = false;
  @Input() error = '';
  @Input() suggestions: AgentResponse | null = null;

  formatCentipawns(value: number): string {
    const pawns = value / 100;
    return pawns > 0 ? `+${pawns.toFixed(2)}` : pawns.toFixed(2);
  }

  hasStats(rec: AgentRecommendation): boolean {
    return (
      rec.averageRating !== undefined ||
      rec.white !== undefined ||
      rec.draws !== undefined ||
      rec.black !== undefined
    );
  }
}
