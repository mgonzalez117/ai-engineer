import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { AgentResponse, AgentRecommendation } from '../service/api/agent.types';

@Component({
  selector: 'app-agent-sidebar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './agent-sidebar.html',
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
