import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

import { ChessboardComponent } from './chessboard/chessboard.component';
import { AgentSidebarComponent } from './agent-sidebar/agent-sidebar';
import { LearningPanelComponent } from './learning-panel/learning-panel';
import { AgentService } from './service/api/AgentService';
import { AgentResponse } from './service/api/agent.types';

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

  constructor(private agentService: AgentService) {}

  onFenChange(fen: string): void {
    this.currentFen = fen;
    this.loading = true;
    this.error = '';

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
}
