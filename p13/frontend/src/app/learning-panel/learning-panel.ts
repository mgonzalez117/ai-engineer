import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

type VideoItem = {
  title: string;
  channel: string;
  url: string;
  thumbnail: string;
};

type ContextItem = {
  id?: string;
  score?: number;
  text: string;
};

@Component({
  selector: 'app-learning-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './learning-panel.html',
  styleUrl: './learning-panel.css',
})
export class LearningPanelComponent {
  @Input() videos: VideoItem[] = [];
  @Input() context: ContextItem[] = [];
}
