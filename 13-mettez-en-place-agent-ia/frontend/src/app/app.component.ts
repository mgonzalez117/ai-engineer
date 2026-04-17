import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

import { ChessboardComponent } from './chessboard/chessboard.component';
import { AgentSidebarComponent } from './agent-sidebar/agent-sidebar';
import { LearningPanelComponent } from './learning-panel/learning-panel';
import { AgentService } from './service/api/AgentService';
import { AgentRecommendation, AgentResponse } from './service/api/agent.types';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    ChessboardComponent,
    AgentSidebarComponent,
    LearningPanelComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  currentFen = '';
  loading = false;
  error = '';
  agentResponse: AgentResponse | null = null;

  /**
   * Case de départ de la pièce actuellement en cours d'interaction.
   * null = pas de filtre temporaire.
   */
  draggedFrom: string | null = null;

  constructor(private agentService: AgentService) {}

  onFenChange(fen: string): void {
    this.currentFen = fen;
    this.loading = true;
    this.error = '';

    /**
     * Dès qu'un coup est réellement joué, on retire le filtre temporaire
     * et on laisse le comportement normal recharger les suggestions.
     */
    this.draggedFrom = null;

    this.agentService.getSuggestions(fen).subscribe({
      next: (data) => {
        this.agentResponse = data;
        this.loading = false;
      },
      error: (err) => {
        console.error('Erreur appel agent', err);
        this.error = 'Impossible de récupérer les suggestions de l’agent.';
        this.agentResponse = null;
        this.loading = false;
      },
    });
  }

  onDragFilterChange(from: string | null): void {
    this.draggedFrom = from;
  }

  get filteredAgentResponse(): AgentResponse | null {
    if (!this.agentResponse) {
      return null;
    }

    if (!this.draggedFrom) {
      return this.agentResponse;
    }

    return {
      ...this.agentResponse,
      recommendations: this.filterRecommendationsByFrom(
        this.agentResponse.recommendations ?? [],
        this.draggedFrom
      ),
    };
  }

  private filterRecommendationsByFrom(
    recommendations: AgentRecommendation[],
    from: string
  ): AgentRecommendation[] {
    return recommendations.filter((recommendation) => {
      const uci = recommendation.uci;

      if (!uci || uci.length < 4) {
        return false;
      }

      return uci.slice(0, 2) === from;
    });
  }
}
